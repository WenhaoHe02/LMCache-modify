# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.kv_object_store import (
    KVObjectByteRange,
    KVObjectId,
    KVObjectMetadataStore,
    KVObjectPoolFullError,
    KVObjectPoolIO,
    KVObjectPoolLayout,
    KVObjectRecord,
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


def test_pool_layout_dense_allocates_by_object_length(tmp_path: Path) -> None:
    layout = KVObjectPoolLayout(
        pool_id="rank3-full",
        pool_path=tmp_path / "rank3_full.pool",
        slot_bytes=8192,
        capacity=2,
        alignment=4096,
        dense=True,
    )
    first = layout.allocate(
        KVObjectId(
            model_id="model",
            parallel_config_id="tp8",
            rank=3,
            layer_id=0,
            role="full",
            block_id="7",
        ),
        length=1024,
        shape=(512, 2),
        dtype="torch.uint8",
    )
    second = layout.allocate(
        KVObjectId(
            model_id="model",
            parallel_config_id="tp8",
            rank=3,
            layer_id=0,
            role="full",
            block_id="8",
        ),
        length=5000,
        shape=(2500, 2),
        dtype="torch.uint8",
    )

    assert first.offset == 0
    assert first.aligned_length == 4096
    assert second.offset == 4096
    assert second.aligned_length == 8192
    assert layout.pool_size_bytes() == 12288
    assert layout.pool_path.stat().st_size == layout.pool_size_bytes()


def test_pool_layout_dense_allows_objects_larger_than_nominal_slot(
    tmp_path: Path,
) -> None:
    layout = KVObjectPoolLayout(
        pool_id="rank3-full",
        pool_path=tmp_path / "rank3_full.pool",
        slot_bytes=4096,
        capacity=2,
        alignment=4096,
        dense=True,
    )

    first = layout.allocate(
        KVObjectId(
            model_id="model",
            parallel_config_id="tp8",
            rank=3,
            layer_id=0,
            role="full",
            block_id="large",
        ),
        length=4097,
        shape=(4097,),
        dtype="torch.uint8",
    )
    second = layout.allocate(
        KVObjectId(
            model_id="model",
            parallel_config_id="tp8",
            rank=3,
            layer_id=0,
            role="full",
            block_id="small",
        ),
        length=16,
        shape=(16,),
        dtype="torch.uint8",
    )

    assert first.offset == 0
    assert first.aligned_length == 8192
    assert second.offset == 8192
    assert layout.pool_size_bytes() == 12288


def test_pool_layout_next_allocation_bounds_do_not_consume_slot(
    tmp_path: Path,
) -> None:
    layout = KVObjectPoolLayout(
        pool_id="rank3-full",
        pool_path=tmp_path / "rank3_full.pool",
        slot_bytes=4096,
        capacity=1,
        alignment=4096,
        dense=True,
    )

    assert layout.next_allocation_bounds(4097) == (0, 8192, 8192)
    assert layout.pool_size_bytes() == 0

    record = layout.allocate(
        KVObjectId(
            model_id="model",
            parallel_config_id="tp8",
            rank=3,
            layer_id=0,
            role="full",
            block_id="large",
        ),
        length=4097,
        shape=(4097,),
        dtype="torch.uint8",
    )

    assert record.offset == 0
    assert layout.pool_size_bytes() == 8192


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


def test_pool_io_reads_and_writes_many_records_in_order(tmp_path: Path) -> None:
    layout = KVObjectPoolLayout(
        pool_id="rank0-csa",
        pool_path=tmp_path / "rank0_csa.pool",
        slot_bytes=4096,
        capacity=3,
    )
    records = [
        layout.allocate(
            KVObjectId(
                model_id="model",
                parallel_config_id="tp8",
                rank=0,
                layer_id=layer_id,
                role="csa",
                block_id=str(layer_id),
            ),
            length=16,
            shape=(8, 2),
            dtype="torch.uint8",
        ).mark_ready()
        for layer_id in range(3)
    ]
    payloads = [
        bytes([index]) * records[index].length for index in range(len(records))
    ]
    pool_io = KVObjectPoolIO({"rank0-csa": layout.pool_path})

    write_ms = pool_io.write_many(records, payloads)
    read_batch = pool_io.read_many([records[2], records[0], records[1]])

    assert write_ms >= 0.0
    assert read_batch.elapsed_ms >= 0.0
    assert read_batch.bytes_read == 48
    assert read_batch.payloads == [payloads[2], payloads[0], payloads[1]]


def test_pool_io_rejects_payload_length_mismatch(tmp_path: Path) -> None:
    layout = KVObjectPoolLayout(
        pool_id="rank0-full",
        pool_path=tmp_path / "rank0_full.pool",
        slot_bytes=4096,
        capacity=1,
    )
    record = layout.allocate(
        KVObjectId(
            model_id="model",
            parallel_config_id="tp8",
            rank=0,
            layer_id=0,
            role="full",
            block_id="0",
        ),
        length=16,
        shape=(8, 2),
        dtype="torch.uint8",
    )
    pool_io = KVObjectPoolIO({"rank0-full": layout.pool_path})

    with pytest.raises(ValueError, match="payload length"):
        pool_io.write_object(record, b"short")


def test_byte_range_round_trips() -> None:
    byte_range = KVObjectByteRange(offset=128, length=64, target_offset=32)
    restored = KVObjectByteRange.from_dict(byte_range.to_dict())

    assert restored == byte_range


def test_record_read_ranges_and_json_round_trips() -> None:
    object_id = KVObjectId(
        model_id="model",
        parallel_config_id="tp8",
        rank=0,
        layer_id=3,
        role="csa_attention_kv",
        block_id="abc",
    )
    record = KVObjectRecord(
        object_id=object_id,
        pool_id="rank0-csa",
        offset=256,
        length=192,
        aligned_length=192,
        shape=(192,),
        dtype="torch.uint8",
        state=KVObjectState.READY,
        byte_ranges=(
            KVObjectByteRange(offset=256, length=64, target_offset=0),
            KVObjectByteRange(offset=320, length=128, target_offset=64),
        ),
    )

    restored = KVObjectRecord.from_dict(record.to_dict())

    assert restored == record
    assert len(record.read_ranges) == 2


def test_record_prefix_view_preserves_logical_prefix_ranges() -> None:
    object_id = KVObjectId(
        model_id="model",
        parallel_config_id="tp8",
        rank=0,
        layer_id=3,
        role="csa_attention_kv",
        block_id="abc",
    )
    record = KVObjectRecord(
        object_id=object_id,
        pool_id="rank0-csa",
        offset=256,
        length=192,
        aligned_length=192,
        shape=(192,),
        dtype="torch.uint8",
        state=KVObjectState.READY,
        byte_ranges=(
            KVObjectByteRange(offset=256, length=64, target_offset=0),
            KVObjectByteRange(offset=512, length=128, target_offset=64),
        ),
    )

    view = record.with_byte_ranges(
        (
            KVObjectByteRange(offset=256, length=64, target_offset=0),
            KVObjectByteRange(offset=512, length=32, target_offset=64),
        ),
        length=96,
    )

    assert view.length == 96
    assert view.offset == 256
    assert view.read_ranges == (
        KVObjectByteRange(offset=256, length=64, target_offset=0),
        KVObjectByteRange(offset=512, length=32, target_offset=64),
    )


def test_record_rejects_gapped_read_ranges() -> None:
    object_id = KVObjectId(
        model_id="model",
        parallel_config_id="tp8",
        rank=0,
        layer_id=3,
        role="csa_attention_kv",
        block_id="abc",
    )

    with pytest.raises(ValueError, match="exactly cover"):
        KVObjectRecord(
            object_id=object_id,
            pool_id="rank0-csa",
            offset=256,
            length=192,
            aligned_length=192,
            shape=(192,),
            dtype="torch.uint8",
            state=KVObjectState.READY,
            byte_ranges=(
                KVObjectByteRange(offset=256, length=64, target_offset=0),
                KVObjectByteRange(offset=320, length=64, target_offset=128),
            ),
        )


def test_pool_io_reads_multi_range_records(tmp_path: Path) -> None:
    layout = KVObjectPoolLayout(
        pool_id="rank0-csa",
        pool_path=tmp_path / "rank0_csa.pool",
        slot_bytes=4096,
        capacity=1,
        dense=True,
    )
    record = (
        layout.allocate(
            KVObjectId(
                model_id="model",
                parallel_config_id="tp8",
                rank=0,
                layer_id=3,
                role="csa_attention_kv",
                block_id="abc",
            ),
            length=192,
            shape=(192,),
            dtype="torch.uint8",
        )
        .with_byte_ranges(
            [
                KVObjectByteRange(offset=0, length=64, target_offset=0),
                KVObjectByteRange(offset=128, length=64, target_offset=64),
                KVObjectByteRange(offset=256, length=64, target_offset=128),
            ]
        )
        .mark_ready()
    )

    with layout.pool_path.open("wb") as handle:
        handle.write(bytes(range(64)))
        handle.write(bytes(64))
        handle.write(bytes(range(64, 128)))
        handle.write(bytes(64))
        handle.write(bytes(range(128, 192)))

    pool_io = KVObjectPoolIO({"rank0-csa": layout.pool_path})
    batch = pool_io.read_many([record])

    assert batch.bytes_read == 192
    assert batch.payloads[0][:4] == bytes(range(4))
    assert batch.payloads[0][64:68] == bytes(range(64, 68))
    assert batch.payloads[0][128:132] == bytes(range(128, 132))
