# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.kv_object_store import (
    KVObjectId,
    KVObjectMetadataStore,
    KVObjectPoolFullError,
    KVObjectPoolLayout,
    KVObjectState,
)


def test_object_id_round_trips_stable_key() -> None:
    object_id = KVObjectId(
        model_id="deepseek-v4-pro",
        parallel_config_id="tp8",
        rank=3,
        layer_id=18,
        role="csa",
        block_id="block-123",
    )

    restored = KVObjectId.from_key(object_id.to_key())

    assert restored == object_id
    assert restored.to_key() == object_id.to_key()


def test_pool_layout_allocates_aligned_sparse_slots(tmp_path: Path) -> None:
    layout = KVObjectPoolLayout(
        pool_id="rank3-csa",
        pool_path=tmp_path / "rank3_csa.pool",
        slot_bytes=6000,
        capacity=2,
        alignment=4096,
    )
    object_id = KVObjectId(
        model_id="model",
        parallel_config_id="tp8",
        rank=3,
        layer_id=4,
        role="csa",
        block_id="7",
    )

    first = layout.allocate(
        object_id,
        length=5000,
        shape=(2, 16, 128),
        dtype="torch.bfloat16",
    )
    second = layout.allocate(
        KVObjectId(
            model_id="model",
            parallel_config_id="tp8",
            rank=3,
            layer_id=4,
            role="csa",
            block_id="8",
        ),
        length=5000,
        shape=(2, 16, 128),
        dtype="torch.bfloat16",
    )

    assert first.offset == 0
    assert first.aligned_length == 8192
    assert second.offset == 8192
    assert layout.pool_path.stat().st_size == layout.pool_size_bytes()


def test_pool_layout_rejects_full_and_oversized_objects(tmp_path: Path) -> None:
    layout = KVObjectPoolLayout(
        pool_id="rank0-hca",
        pool_path=tmp_path / "rank0_hca.pool",
        slot_bytes=4096,
        capacity=1,
    )
    object_id = KVObjectId(
        model_id="model",
        parallel_config_id="tp8",
        rank=0,
        layer_id=0,
        role="hca",
        block_id="0",
    )

    with pytest.raises(ValueError, match="exceeds"):
        layout.allocate(
            object_id,
            length=4097,
            shape=(4097,),
            dtype="torch.uint8",
        )

    layout.allocate(
        object_id,
        length=4096,
        shape=(4096,),
        dtype="torch.uint8",
    )
    with pytest.raises(KVObjectPoolFullError):
        layout.allocate(
            KVObjectId(
                model_id="model",
                parallel_config_id="tp8",
                rank=0,
                layer_id=0,
                role="hca",
                block_id="1",
            ),
            length=4096,
            shape=(4096,),
            dtype="torch.uint8",
        )


def test_metadata_store_ready_lookup_and_jsonl_roundtrip(tmp_path: Path) -> None:
    store = KVObjectMetadataStore()
    object_id = KVObjectId(
        model_id="model",
        parallel_config_id="tp8",
        rank=1,
        layer_id=2,
        role="full",
        block_id="42",
    )
    layout = KVObjectPoolLayout(
        pool_id="rank1-full",
        pool_path=tmp_path / "rank1_full.pool",
        slot_bytes=4096,
        capacity=1,
    )
    allocated = layout.allocate(
        object_id,
        length=1024,
        shape=(512, 2),
        dtype="torch.bfloat16",
    )

    store.put(allocated)
    assert store.get_many([object_id]) == [None]

    ready = allocated.mark_ready()
    store.put(ready)
    assert store.get(object_id) == ready
    assert store.get_many([object_id]) == [ready]
    assert ready.state == KVObjectState.READY

    metadata_path = tmp_path / "metadata.jsonl"
    store.dump_jsonl(metadata_path)
    restored = KVObjectMetadataStore.load_jsonl(metadata_path)

    assert restored.get(object_id) == ready
