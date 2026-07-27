# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import json
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc.transport import (
    RpcClientTransport,
    RpcServerTransport,
)

logger = init_logger(__name__)


def _env_flag(name: str) -> bool:
    """Return whether an environment variable contains a truthy value."""
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


class LMCacheLookupClient(LookupClientInterface):
    """
    Lookup client that communicates with a lookup server
    via an injected RpcClientTransport.

    The client is decoupled from the underlying communication
    mechanism. The transport layer handles connection management,
    retries, and error recovery.

    Related extra_config:
    - lookup_server_worker_ids:
        is a config to control create lookup server on some
        workers.
        if mla is not enabled, default is [];
        if mla is enabled, default is [0];
        - if lookup_server_worker_ids is [], start lookup
          server on all workers
        - if lookup_server_worker_ids is [0], start lookup
          server on worker0
        - if lookup_server_worker_ids is [0, 3, 6], start
          lookup server on worker0, worker3 and worker6
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        transport: RpcClientTransport,
    ):
        self.config = config
        self.transport = transport

        # NOTE: map from lookup_id (i.e., req_id) to
        # req's status.
        # int indicates number of hit tokens.
        # The assumption here is that once a request is
        # looked up, the following lookups of the same
        # request must have the same result.
        self.reqs_status: dict[str, int] = {}
        self.reqs_terminal_hash: dict[str, int] = {}
        self._recent_prefix_lock = threading.Lock()
        self._recent_prefix_token_count = 0
        self._recent_prefix_hashes: list[int] = []

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.enable_blending = config.enable_blending
        self.token_database: TokenDatabase
        if self.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        "-1 means not found;
        None means ongoing; (not supported in sync client)
        int >= 0 means number of hit tokens
        """
        return self.reqs_status.get(lookup_id, -1)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        lookup_start = time.perf_counter()
        request_configs_str = ""
        if request_configs is not None and len(request_configs) != 0:
            request_configs_str = json.dumps(request_configs)

        # NOTE(Jiayi): We cannot only send hashes when
        # blending enabled because the blender need the
        # input embedding.
        if not self.enable_blending:
            hashes = []
            offsets = []

            for (
                start,
                end,
                key,
            ) in self.token_database.process_tokens(token_ids, make_key=False):
                hashes.append(key)
                offsets.append(end - start)

            # if the token database returns no hashes,
            # return 0
            if not hashes:
                return 0

            hash_done = time.perf_counter()
            terminal_hint = self._streaming_terminal_hint(
                token_ids,
                hashes,
                offsets,
            )

            if os.getenv("LMCACHE_LOOKUP_TOKEN_DIAGNOSTICS", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                first_token = token_ids[0] if len(token_ids) else None
                logger.info(
                    "LMCache lookup hash diagnostic req=%s token_container=%s "
                    "token_type=%s chunk_size=%d first_hash=%s last_hash=%s "
                    "chunks=%d",
                    lookup_id,
                    type(token_ids).__name__,
                    type(first_token).__name__ if first_token is not None else "none",
                    self.config.chunk_size,
                    hashes[0],
                    hashes[-1],
                    len(hashes),
                )

            msg_buf = [hashes, offsets]
            if terminal_hint is not None:
                msg_buf.append(terminal_hint)
            msg_buf.extend([lookup_id, request_configs_str])
        else:
            # Convert token_ids to a plain list for msgpack serialization
            # (vLLM 0.18+ may pass ConstantList which msgspec can't encode)
            if isinstance(token_ids, torch.Tensor):
                serializable_ids = token_ids.tolist()
            elif not isinstance(token_ids, list):
                serializable_ids = list(token_ids)
            else:
                serializable_ids = token_ids
            msg_buf = [
                serializable_ids,
                lookup_id,
                request_configs_str,
            ]

        rpc_start = time.perf_counter()
        responses = self.transport.send_and_recv_all(msg_buf)
        rpc_done = time.perf_counter()

        # Transport returns empty list on failure
        if not responses:
            return 0

        results = [int.from_bytes(resp, "big") for resp in responses]

        assert len(results) == self.transport.world_size
        if len(set(results)) > 1:
            logger.warning(
                "Lookup results (number of hit tokens) "
                "differ across (TP and PP) ranks: %s.",
                results,
            )
        # NOTE: it is possible that the number of hit
        # tokens is different across (TP and PP) ranks,
        # so we can use the minimum value.
        num_hit_toks = min(results)
        self.reqs_status[lookup_id] = num_hit_toks
        self.reqs_terminal_hash.pop(lookup_id, None)
        if not self.enable_blending:
            self._update_recent_prefix(hashes, offsets, num_hit_toks)
            hit_chunks = num_hit_toks // self.config.chunk_size
            if (
                hit_chunks > 0
                and num_hit_toks % self.config.chunk_size == 0
                and hit_chunks <= len(hashes)
            ):
                self.reqs_terminal_hash[lookup_id] = int(hashes[hit_chunks - 1])

        if _env_flag("LMCACHE_TTFT_STAGE_PROFILE"):
            logger.info(
                "LMCACHE_TTFT_LOOKUP req=%s hashes=%d terminal_hint=%d "
                "hash_ms=%.3f rpc_ms=%.3f total_ms=%.3f",
                lookup_id,
                len(hashes) if not self.enable_blending else 0,
                int(terminal_hint is not None) if not self.enable_blending else 0,
                (
                    (hash_done - lookup_start) * 1000.0
                    if not self.enable_blending
                    else 0.0
                ),
                (rpc_done - rpc_start) * 1000.0,
                (time.perf_counter() - lookup_start) * 1000.0,
            )

        return num_hit_toks

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.reqs_status.pop(lookup_id, None)
        self.reqs_terminal_hash.pop(lookup_id, None)

    def lookup_terminal_hash(self, lookup_id: str) -> Optional[int]:
        """Return the final hit chunk hash cached by :meth:`lookup`.

        Args:
            lookup_id: Lookup identifier previously passed to :meth:`lookup`.

        Returns:
            The terminal chunk hash for an aligned non-empty hit, otherwise
            ``None``.
        """
        return self.reqs_terminal_hash.get(lookup_id)

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports
        producer kvcache reuse"""
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.transport.close()

    def _streaming_terminal_hint(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        hashes: list[int],
        offsets: list[int],
    ) -> Optional[dict[str, int | str]]:
        """Return a verified recent-prefix hint for ON streaming lookup."""
        if not _env_flag("LMCACHE_INDEXER_ENABLE_PREFETCH"):
            return None
        with self._recent_prefix_lock:
            candidate_tokens = self._recent_prefix_token_count
            candidate_chunks = len(self._recent_prefix_hashes)
            if (
                candidate_tokens == 0
                or candidate_tokens % self.config.chunk_size != 0
                or candidate_chunks != candidate_tokens // self.config.chunk_size
                or candidate_tokens > len(token_ids)
                or candidate_chunks > len(hashes)
                or int(hashes[candidate_chunks - 1])
                != int(self._recent_prefix_hashes[-1])
            ):
                return None
            current_hashes = [int(value) for value in hashes[:candidate_chunks]]
            if current_hashes != self._recent_prefix_hashes:
                return None
            return {
                "mode": "streaming_terminal_prefix",
                "terminal_hash": int(self._recent_prefix_hashes[-1]),
                "tokens": candidate_tokens,
            }

    def _update_recent_prefix(
        self,
        hashes: list[int],
        offsets: list[int],
        num_hit_toks: int,
    ) -> None:
        """Cache one aligned prefix for the next related service request."""
        full_chunks = 0
        for offset in offsets:
            if int(offset) != self.config.chunk_size:
                break
            full_chunks += 1
        reusable_tokens = full_chunks * self.config.chunk_size
        aligned_hit_tokens = (
            num_hit_toks // self.config.chunk_size * self.config.chunk_size
        )
        if aligned_hit_tokens > 0:
            reusable_tokens = min(reusable_tokens, aligned_hit_tokens)
            full_chunks = reusable_tokens // self.config.chunk_size
        if reusable_tokens <= 0 or full_chunks <= 0:
            return
        with self._recent_prefix_lock:
            self._recent_prefix_token_count = reusable_tokens
            self._recent_prefix_hashes = [int(value) for value in hashes[:full_chunks]]


class LMCacheLookupServer:
    """Lookup server that handles lookup requests using
    LMCacheEngine, with an injected RpcServerTransport."""

    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        metadata: LMCacheMetadata,
        transport: RpcServerTransport,
    ):
        self.transport = transport
        self.lmcache_engine = lmcache_engine
        self.running = True
        self.enable_blending = lmcache_engine.config.enable_blending

        def process_request():
            while self.running:
                try:
                    result = self.transport.recv_request()
                    if result is None:
                        continue

                    identity, data_frames = result

                    # Validate frame structure
                    if len(data_frames) < 3:
                        logger.warning("Malformed request received: not enough frames.")
                        continue

                    # Validate and decode lookup_id
                    lookup_id_bytes = data_frames[-2]
                    request_configs_bytes = data_frames[-1]

                    if not isinstance(lookup_id_bytes, (bytes, str)):
                        logger.warning(
                            "Malformed request received: lookup_id is not bytes or str."
                        )
                        continue

                    if not isinstance(request_configs_bytes, (bytes, str)):
                        logger.warning(
                            "Malformed request received: "
                            "request_configs is not bytes or str."
                        )
                        continue

                    # Decode to strings
                    if isinstance(lookup_id_bytes, bytes):
                        lookup_id = lookup_id_bytes.decode("utf-8")
                    else:
                        lookup_id = lookup_id_bytes

                    if isinstance(request_configs_bytes, bytes):
                        request_configs_str = request_configs_bytes.decode("utf-8")
                    else:
                        request_configs_str = request_configs_bytes

                    request_configs = (
                        json.loads(request_configs_str) if request_configs_str else None
                    )

                    if not self.enable_blending:
                        hashes = data_frames[0]
                        offsets = data_frames[1]
                        terminal_hint = (
                            data_frames[-3]
                            if len(data_frames) >= 5
                            and isinstance(data_frames[-3], dict)
                            else None
                        )
                        if os.getenv(
                            "LMCACHE_LOOKUP_TOKEN_DIAGNOSTICS", "0"
                        ).lower() in {"1", "true", "yes", "on"}:
                            logger.info(
                                "LMCache lookup server hash diagnostic req=%s "
                                "first_hash=%s last_hash=%s chunks=%d",
                                lookup_id,
                                hashes[0] if hashes else None,
                                hashes[-1] if hashes else None,
                                len(hashes),
                            )
                        fast_start = time.perf_counter()
                        lookup_result: Optional[int] = None
                        terminal_fast_hit = False
                        if (
                            (lookup_result is None or lookup_result == 0)
                            and terminal_hint is not None
                            and terminal_hint.get("mode")
                            == "streaming_terminal_prefix"
                        ):
                            candidate_tokens = int(terminal_hint.get("tokens", 0))
                            candidate_chunks = (
                                candidate_tokens
                                // self.lmcache_engine.config.chunk_size
                            )
                            candidate_hash = int(
                                terminal_hint.get("terminal_hash", 0)
                            )
                            hint_valid = bool(
                                candidate_tokens > 0
                                and candidate_tokens
                                % self.lmcache_engine.config.chunk_size
                                == 0
                                and candidate_chunks <= len(hashes)
                                and candidate_chunks <= len(offsets)
                                and int(offsets[candidate_chunks - 1])
                                == self.lmcache_engine.config.chunk_size
                                and int(hashes[candidate_chunks - 1])
                                == candidate_hash
                            )
                            if hint_valid:
                                candidate_result = (
                                    self.lmcache_engine.lookup_streaming_terminal(
                                        terminal_hash=candidate_hash,
                                        token_count=candidate_tokens,
                                        lookup_id=lookup_id,
                                        pin=True,
                                        request_configs=request_configs,
                                    )
                                )
                                if candidate_result == candidate_tokens:
                                    lookup_result = candidate_result
                                    terminal_fast_hit = True
                        if _env_flag("LMCACHE_INDEXER_ENABLE_PREFETCH"):
                            # Admission publishes one immutable layer-major
                            # generation per cached prefix.  The current
                            # request normally appends a short recompute suffix,
                            # so probe terminal generations from the end toward
                            # the beginning before falling back to ordinary
                            # per-chunk lookup.  This makes the *first* hit after
                            # cold admission discoverable; relying on a previous
                            # hit hint creates a 0-hit/short-hit fixed point.
                            full_chunk_indices: list[int] = []
                            for index, offset in enumerate(offsets):
                                if (
                                    int(offset)
                                    != self.lmcache_engine.config.chunk_size
                                ):
                                    break
                                full_chunk_indices.append(index)
                            for candidate_index in (
                                reversed(full_chunk_indices)
                                if lookup_result is None or lookup_result == 0
                                else ()
                            ):
                                candidate_tokens = (
                                    candidate_index + 1
                                ) * self.lmcache_engine.config.chunk_size
                                candidate_result = (
                                    self.lmcache_engine.lookup_streaming_terminal(
                                        terminal_hash=int(hashes[candidate_index]),
                                        token_count=candidate_tokens,
                                        lookup_id=lookup_id,
                                        pin=True,
                                        request_configs=request_configs,
                                    )
                                )
                                if candidate_result == candidate_tokens:
                                    lookup_result = candidate_result
                                    terminal_fast_hit = True
                                    break
                        fast_done = time.perf_counter()
                        if lookup_result is None:
                            lookup_result = self.lmcache_engine.lookup(
                                hashes=hashes,
                                offsets=offsets,
                                lookup_id=lookup_id,
                                pin=True,
                                request_configs=request_configs,
                            )
                        if _env_flag("LMCACHE_TTFT_STAGE_PROFILE"):
                            logger.info(
                                "LMCACHE_TTFT_LOOKUP_SERVER req=%s hashes=%d "
                                "terminal_hint=%d terminal_hit=%d fast_ms=%.3f "
                                "total_ms=%.3f",
                                lookup_id,
                                len(hashes),
                                int(terminal_hint is not None),
                                int(terminal_fast_hit),
                                (fast_done - fast_start) * 1000.0,
                                (time.perf_counter() - fast_start) * 1000.0,
                            )
                    else:
                        tokens = data_frames[0]
                        lookup_result = self.lmcache_engine.lookup(
                            tokens=tokens,
                            lookup_id=lookup_id,
                            pin=True,
                            request_configs=request_configs,
                        )
                    response = lookup_result.to_bytes(4, "big")
                    self.transport.send_response(identity, response)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON in lookup request: {e}")
                except UnicodeDecodeError as e:
                    logger.error(f"Error decoding UTF-8 in lookup request: {e}")
                except Exception:
                    logger.exception("Error processing lookup request")

        logger.info("lmcache lookup server started")
        self.thread = threading.Thread(
            target=process_request,
            daemon=True,
            name="lookup-server-thread",
        )
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        # Stop the processing thread first
        self.running = False

        # Wait for thread to finish with timeout
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Lookup server thread did not terminate gracefully")

        # Close transport after thread is stopped
        self.transport.close()
