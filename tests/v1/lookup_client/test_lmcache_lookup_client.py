# SPDX-License-Identifier: Apache-2.0
"""
Tests for LMCacheLookupClient and LMCacheLookupServer communication.

IMPORTANT: These tests require PYTHONHASHSEED to be set for consistent hashing.
In production, LMCacheLookupClient and LMCacheLookupServer run in separate
processes (scheduler vs worker), and PYTHONHASHSEED must be set consistently
across all processes to ensure hash consistency for cache lookups.

Run with: PYTHONHASHSEED=0 pytest tests/v1/lookup_client/test_lmcache_lookup_client.py
"""

# Standard
import os
import random
import tempfile
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.lookup_client.factory import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_lookup_client import (
    LMCacheLookupClient,
    LMCacheLookupServer,
)
from tests.v1.utils import (
    create_test_config,
    create_test_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
    recover_engine_states,
)

# Skip all tests in this module if PYTHONHASHSEED is not set.
# This reflects production requirements where consistent hashing across
# processes (scheduler/worker) requires PYTHONHASHSEED to be set.
pytestmark = pytest.mark.skipif(
    os.environ.get("PYTHONHASHSEED") is None,
    reason=(
        "PYTHONHASHSEED must be set for consistent hashing between "
        "LMCacheLookupClient and LMCacheLookupServer. "
        "Run with: PYTHONHASHSEED=0 pytest ..."
    ),
)


class TestLMCacheLookupClientServer:
    """Test suite for LMCacheLookupClient and LMCacheLookupServer communication."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir

    @pytest.fixture
    def lmcache_engine_metadata(self):
        """Create test metadata for LMCacheEngine."""
        return create_test_metadata()

    @pytest.fixture
    def lmcache_engine(self, temp_dir, lmcache_engine_metadata):
        """Create a LMCacheEngine instance for testing."""
        instance_id = f"test_lookup_instance_{uuid.uuid4().hex[:8]}"
        config = create_test_config(instance_id=instance_id)

        # Use mock connector for CPU testing
        connector = MockGPUConnector(kv_shape=(32, 2, 256, 8, 128))

        engine = LMCacheEngineBuilder.get_or_create(
            instance_id=instance_id,
            config=config,
            metadata=lmcache_engine_metadata,
            gpu_connector=connector,
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
        )
        engine.post_init()

        yield engine

        # Cleanup
        engine.close()
        # Remove from singleton cache to avoid test pollution
        LMCacheEngineBuilder._instances.pop(instance_id, None)
        LMCacheEngineBuilder._cfgs.pop(instance_id, None)
        LMCacheEngineBuilder._metadatas.pop(instance_id, None)
        LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)

    def _create_server(self, lmcache_engine):
        """Helper to create a lookup server with transport."""
        transport = LookupClientFactory._create_zmq_server_transport(
            lmcache_engine.metadata
        )
        return LMCacheLookupServer(lmcache_engine, lmcache_engine.metadata, transport)

    def _create_client(self, lmcache_engine):
        """Helper to create a lookup client with transport."""
        transport = LookupClientFactory._create_zmq_client_transport(
            lmcache_engine.config, lmcache_engine.metadata
        )
        return LMCacheLookupClient(
            lmcache_engine.config,
            lmcache_engine.metadata,
            transport,
        )

    def test_terminal_hash_is_cached_without_rpc_server(
        self,
        lmcache_engine_metadata,
    ) -> None:
        """The synchronous lookup retains and clears its final hit hash."""
        config = create_test_config(instance_id="terminal_hash_test")
        transport = SimpleNamespace(
            world_size=1,
            send_and_recv_all=lambda _message: [(512).to_bytes(8, "big")],
        )
        client = LMCacheLookupClient(
            config,
            lmcache_engine_metadata,
            transport,
        )
        tokens = generate_tokens(512, "cpu", fixed=True).tolist()
        expected_hash = list(
            client.token_database.process_tokens(tokens, make_key=False)
        )[-1][2]

        assert client.lookup(tokens, "request") == 512
        assert client.lookup_terminal_hash("request") == expected_hash
        client.clear_lookup_status("request")
        assert client.lookup_terminal_hash("request") is None

    def test_related_request_sends_verified_streaming_terminal_hint(
        self,
        lmcache_engine_metadata,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cold lookup seeds a content-verified hint for the first hit."""
        monkeypatch.setenv("LMCACHE_INDEXER_ENABLE_PREFETCH", "1")
        messages: list[list[object]] = []
        responses = [0, 512]

        def send_and_recv(message: list[object]) -> list[bytes]:
            messages.append(message)
            return [responses.pop(0).to_bytes(4, "big")]

        config = create_test_config(instance_id="streaming_terminal_hint")
        transport = SimpleNamespace(
            world_size=1,
            send_and_recv_all=send_and_recv,
        )
        client = LMCacheLookupClient(config, lmcache_engine_metadata, transport)
        base_tokens = generate_tokens(512, "cpu", fixed=True).tolist()
        extended_tokens = base_tokens + [100_000 + index for index in range(256)]

        assert client.lookup(base_tokens, "cold") == 0
        assert client.lookup(extended_tokens, "hit") == 512

        assert len(messages[0]) == 4
        assert len(messages[1]) == 5
        hint = messages[1][2]
        assert isinstance(hint, dict)
        assert hint["mode"] == "streaming_terminal_prefix"
        assert hint["tokens"] == 512
        assert hint["terminal_hash"] == messages[0][0][-1]

    @pytest.mark.parametrize("terminal_hit", [True, False])
    def test_server_streaming_terminal_fastpath_and_fallback(
        self,
        terminal_hit: bool,
    ) -> None:
        """The server uses exact manifest lookup and falls back on a miss."""

        class FakeTransport:
            def __init__(self) -> None:
                self.requests = [
                    (
                        b"identity",
                        [
                            [11, 22, 33],
                            [256, 256, 256],
                            {
                                "mode": "streaming_terminal_prefix",
                                "terminal_hash": 22,
                                "tokens": 512,
                            },
                            "request",
                            "",
                        ],
                    )
                ]
                self.response: Optional[bytes] = None
                self.response_ready = threading.Event()

            def recv_request(self):
                if self.requests:
                    return self.requests.pop(0)
                time.sleep(0.001)
                return None

            def send_response(self, _identity: bytes, response: bytes) -> None:
                self.response = response
                self.response_ready.set()

            def close(self) -> None:
                return None

        class FakeEngine:
            def __init__(self) -> None:
                self.config = SimpleNamespace(enable_blending=False, chunk_size=256)
                self.calls: list[tuple[list[int], list[int]]] = []
                self.terminal_calls: list[tuple[int, int, str]] = []

            def lookup_streaming_terminal(
                self,
                *,
                terminal_hash: int,
                token_count: int,
                lookup_id: str,
                **_kwargs,
            ) -> int:
                self.terminal_calls.append((terminal_hash, token_count, lookup_id))
                return token_count if terminal_hit else 0

            def lookup(self, *, hashes, offsets, **_kwargs) -> int:
                self.calls.append((hashes, offsets))
                return 512

        transport = FakeTransport()
        engine = FakeEngine()
        server = LMCacheLookupServer(engine, SimpleNamespace(), transport)
        try:
            assert transport.response_ready.wait(1.0)
        finally:
            server.close()

        assert int.from_bytes(transport.response, "big") == 512
        assert engine.terminal_calls == [(22, 512, "request")]
        if terminal_hit:
            assert engine.calls == []
        else:
            assert engine.calls == [([11, 22, 33], [256, 256, 256])]

    def test_basic_lookup_communication(self, lmcache_engine):
        """Test basic lookup communication between client and server."""
        device = "cpu"
        num_tokens = 512
        num_blocks = 100
        block_size = 16

        # Prepare test data
        tokens = generate_tokens(num_tokens, device, fixed=True)
        kv_cache = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
        slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
        slot_mapping = torch.tensor(slot_mapping, device=device)

        # Store data into cache
        lmcache_engine.store(
            tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping
        )
        recover_engine_states(lmcache_engine)
        time.sleep(0.5)

        with self._create_server(lmcache_engine):
            time.sleep(0.5)
            with self._create_client(lmcache_engine) as client:
                lookup_id = "test_request_1"
                result = client.lookup(tokens.tolist(), lookup_id)

                # Verify exact match
                assert result == num_tokens, f"Expected {num_tokens}, got {result}"

                # Verify lookup status is cached
                cached_result = client.lookup_cache(lookup_id)
                assert cached_result == num_tokens
                expected_terminal_hash = list(
                    client.token_database.process_tokens(
                        tokens.tolist(),
                        make_key=False,
                    )
                )[-1][2]
                assert client.lookup_terminal_hash(lookup_id) == expected_terminal_hash

                # Test clear lookup status
                client.clear_lookup_status(lookup_id)
                assert client.lookup_cache(lookup_id) == -1
                assert client.lookup_terminal_hash(lookup_id) is None

                # Test supports_producer_reuse
                assert client.supports_producer_reuse() is True

    def test_multiple_lookups(self, lmcache_engine):
        """Test multiple lookup requests."""
        device = "cpu"
        num_blocks = 200
        block_size = 16

        # Store multiple token sequences
        stored_tokens = []
        for i in range(5):
            num_tokens = 256
            tokens = generate_tokens(num_tokens, device, fixed=True)
            # Make each sequence unique by adding offset
            tokens = tokens + i * 10000
            kv_cache = generate_kv_cache_paged_list_tensors(
                num_blocks, device, block_size
            )
            slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
            slot_mapping = torch.tensor(slot_mapping, device=device)

            lmcache_engine.store(
                tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping
            )
            recover_engine_states(lmcache_engine)
            stored_tokens.append(tokens)

        time.sleep(0.5)

        with self._create_server(lmcache_engine):
            time.sleep(0.5)
            with self._create_client(lmcache_engine) as client:
                # Perform multiple lookups
                for i, tokens in enumerate(stored_tokens):
                    lookup_id = f"test_request_{i}"
                    result = client.lookup(tokens.tolist(), lookup_id)
                    assert result == 256, f"Expected 256, got {result}"
                    assert client.lookup_cache(lookup_id) == 256

    def test_lookup_with_request_configs(self, lmcache_engine):
        """Test lookup with request configurations and tag-based cache isolation."""
        device = "cpu"
        num_tokens = 256
        num_blocks = 100
        block_size = 16

        # Prepare test data for user_a
        tokens_user_a = generate_tokens(num_tokens, device, fixed=True)
        kv_cache = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
        slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
        slot_mapping = torch.tensor(slot_mapping, device=device)

        # Store data with tag: user=user_a
        request_configs_user_a = {
            "temperature": 0.8,
            "top_p": 0.9,
            "lmcache.tag.user": "user_a",
        }
        lmcache_engine.store(
            tokens=tokens_user_a,
            kvcaches=kv_cache,
            slot_mapping=slot_mapping,
            request_configs=request_configs_user_a,
        )
        recover_engine_states(lmcache_engine)
        time.sleep(0.5)

        with self._create_server(lmcache_engine):
            time.sleep(0.5)
            with self._create_client(lmcache_engine) as client:
                # Test 1: Lookup with same tag (user_a) should hit cache
                lookup_id_1 = "test_user_a_match"
                result_1 = client.lookup(
                    tokens_user_a.tolist(), lookup_id_1, request_configs_user_a
                )
                assert result_1 == num_tokens, (
                    f"Expected cache hit with {num_tokens} tokens "
                    f"for user_a, got {result_1}"
                )

                # Test 2: Lookup with different tag (user_b) should NOT hit cache
                request_configs_user_b = {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "lmcache.tag.user": "user_b",
                }
                lookup_id_2 = "test_user_b_no_match"
                result_2 = client.lookup(
                    tokens_user_a.tolist(), lookup_id_2, request_configs_user_b
                )
                assert result_2 == 0, (
                    f"Expected cache miss (0) for user_b, got {result_2}"
                )

                # Test 3: Lookup without tag should NOT hit cache
                request_configs_no_tag = {"temperature": 0.8, "top_p": 0.9}
                lookup_id_3 = "test_no_tag_no_match"
                result_3 = client.lookup(
                    tokens_user_a.tolist(), lookup_id_3, request_configs_no_tag
                )
                assert result_3 == 0, (
                    f"Expected cache miss (0) without tag, got {result_3}"
                )

                # Test 4: Lookup with same tag again should still hit cache
                lookup_id_4 = "test_user_a_match_again"
                result_4 = client.lookup(
                    tokens_user_a.tolist(), lookup_id_4, request_configs_user_a
                )
                assert result_4 == num_tokens, (
                    f"Expected cache hit with {num_tokens} tokens "
                    f"for user_a again, got {result_4}"
                )

                # Test 5: Multiple tags - store with user=user_a and env=prod
                request_configs_multi_tags = {
                    "lmcache.tag.user": "user_a",
                    "lmcache.tag.env": "prod",
                }
                tokens_multi = generate_tokens(num_tokens, device, fixed=True) + 50000
                kv_cache_multi = generate_kv_cache_paged_list_tensors(
                    num_blocks, device, block_size
                )
                slot_mapping_multi = random.sample(
                    range(0, num_blocks * block_size), num_tokens
                )
                slot_mapping_multi = torch.tensor(slot_mapping_multi, device=device)

                lmcache_engine.store(
                    tokens=tokens_multi,
                    kvcaches=kv_cache_multi,
                    slot_mapping=slot_mapping_multi,
                    request_configs=request_configs_multi_tags,
                )
                recover_engine_states(lmcache_engine)
                time.sleep(0.5)

                # Should hit with exact same tags
                lookup_id_5 = "test_multi_tags_match"
                result_5 = client.lookup(
                    tokens_multi.tolist(), lookup_id_5, request_configs_multi_tags
                )
                assert result_5 == num_tokens, (
                    f"Expected cache hit with {num_tokens} tokens "
                    f"for multi tags, got {result_5}"
                )

                # Should NOT hit with partial tags
                request_configs_partial = {"lmcache.tag.user": "user_a"}
                lookup_id_6 = "test_partial_tags_no_match"
                result_6 = client.lookup(
                    tokens_multi.tolist(), lookup_id_6, request_configs_partial
                )
                assert result_6 == 0, (
                    f"Expected cache miss (0) with partial tags, got {result_6}"
                )

                # Should NOT hit with different env tag
                request_configs_diff_env = {
                    "lmcache.tag.user": "user_a",
                    "lmcache.tag.env": "dev",
                }
                lookup_id_7 = "test_diff_env_no_match"
                result_7 = client.lookup(
                    tokens_multi.tolist(), lookup_id_7, request_configs_diff_env
                )
                assert result_7 == 0, (
                    f"Expected cache miss (0) with different env tag, got {result_7}"
                )

    def test_client_timeout_handling(self, lmcache_engine):
        """Test client timeout handling when server is not responding."""
        server = self._create_server(lmcache_engine)
        time.sleep(0.5)

        with self._create_client(lmcache_engine) as client:
            # Close server to simulate timeout
            server.close()
            time.sleep(0.5)

            # Try lookup - should handle timeout gracefully
            token_ids = list(range(256))
            lookup_id = "test_timeout"

            result = client.lookup(token_ids, lookup_id)

            # Should return 0 on timeout
            assert result == 0

    def test_socket_recreation_on_error(self, lmcache_engine):
        """Test socket recreation when ZMQ error occurs."""
        device = "cpu"
        num_tokens = 256
        num_blocks = 100
        block_size = 16

        # Store some data first
        tokens = generate_tokens(num_tokens, device, fixed=True)
        kv_cache = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
        slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
        slot_mapping = torch.tensor(slot_mapping, device=device)

        lmcache_engine.store(
            tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping
        )
        recover_engine_states(lmcache_engine)
        time.sleep(0.5)

        with self._create_server(lmcache_engine) as server:
            time.sleep(0.5)

            with self._create_client(lmcache_engine) as client:
                # First lookup - should hit cache
                token_ids = tokens.tolist()
                result1 = client.lookup(token_ids, "test_1")
                assert result1 == num_tokens

                # Simulate error by closing server
                server.close()
                time.sleep(0.5)

                # This should trigger socket recreation and return 0 on error
                result2 = client.lookup(token_ids, "test_2")
                assert result2 == 0

                # Recreate server
                with self._create_server(lmcache_engine):
                    time.sleep(0.5)

                    # Should work again after socket recreation and hit cache
                    result3 = client.lookup(token_ids, "test_3")
                    assert result3 == num_tokens

    def test_close_methods(self, lmcache_engine):
        """Test proper cleanup of client and server close methods."""
        with self._create_server(lmcache_engine) as server:
            time.sleep(0.5)

            with self._create_client(lmcache_engine) as client:
                # Perform a lookup
                token_ids = list(range(256))
                result = client.lookup(token_ids, "test_close")
                assert result is not None

            # After exiting context, transport is closed

        # After exiting context, verify server thread is stopped
        assert server.running is False
        assert not server.thread.is_alive()

    def test_concurrent_lookups(self, lmcache_engine):
        """Test concurrent lookup requests from same client."""
        with self._create_server(lmcache_engine):
            time.sleep(0.5)
            with self._create_client(lmcache_engine) as client:
                # Perform rapid consecutive lookups
                results = []
                for i in range(10):
                    token_ids = list(range(256))
                    lookup_id = f"concurrent_test_{i}"
                    result = client.lookup(token_ids, lookup_id)
                    results.append(result)

                # All lookups should succeed
                assert len(results) == 10
                assert all(isinstance(r, int) for r in results)

    def test_empty_token_lookup(self, lmcache_engine):
        """Test lookup with empty token list."""
        with self._create_server(lmcache_engine):
            time.sleep(0.5)
            with self._create_client(lmcache_engine) as client:
                # Empty token list
                token_ids = []
                lookup_id = "test_empty"

                result = client.lookup(token_ids, lookup_id)
                assert result is not None
                assert result == 0  # No tokens to lookup

    def test_large_token_lookup(self, lmcache_engine):
        """Test lookup with large number of tokens."""
        device = "cpu"
        num_tokens = 2048
        num_blocks = 500
        block_size = 16

        # Store large token sequence
        tokens = generate_tokens(num_tokens, device, fixed=True)
        kv_cache = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
        slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
        slot_mapping = torch.tensor(slot_mapping, device=device)

        lmcache_engine.store(
            tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping
        )
        recover_engine_states(lmcache_engine)
        time.sleep(0.5)

        with self._create_server(lmcache_engine):
            time.sleep(0.5)
            with self._create_client(lmcache_engine) as client:
                lookup_id = "test_large"
                result = client.lookup(tokens.tolist(), lookup_id)
                assert result == num_tokens, f"Expected {num_tokens}, got {result}"
