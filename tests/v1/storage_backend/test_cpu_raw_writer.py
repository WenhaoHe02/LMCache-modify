# SPDX-License-Identifier: Apache-2.0

# Standard
import os
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.cpu_raw_writer import CPURawBlockWriter


def test_cpu_raw_writer_writes_fragmented_extents_and_zero_fills_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public write API preserves extent order and aligned tail bytes."""
    writes: list[tuple[bytes, int]] = []
    admissions: list[int] = []

    monkeypatch.setattr(os, "O_DIRECT", 0x4000, raising=False)
    monkeypatch.setattr(
        os,
        "pwrite",
        lambda _fd, data, offset: _pwrite(data, offset),
        raising=False,
    )

    def _pwrite(data: memoryview, offset: int) -> int:
        writes.append((bytes(data), offset))
        return len(data)

    with (
        patch("os.open", return_value=91) as open_mock,
        patch("os.close") as close_mock,
    ):
        writer = CPURawBlockWriter(
            "/dev/snvme3n1",
            target_mib_s=0,
            block_bytes=1024,
            wait_for_admission=lambda nbytes, _started: admissions.append(nbytes),
        )
        normalized, elapsed_ms = writer.write(
            bytes(range(256)) * 8,
            raw_extents=[
                (8192, 100, 2),
                (9216, 300, 4),
            ],
            base_file_offset=8192,
            logical_nbytes=3072,
        )
        writer.close()
        writer.close()

    assert normalized == ((8192, 100, 2), (9216, 300, 4))
    assert elapsed_ms >= 0
    assert admissions == [1024, 1024, 1024]
    assert [offset for _, offset in writes] == [51200, 153600, 154624]
    assert writes[0][0] == bytes(range(256)) * 4
    assert writes[1][0] == bytes(range(256)) * 4
    assert writes[2][0] == bytes(1024)
    open_mock.assert_called_once_with("/dev/snvme3n1", os.O_WRONLY | os.O_DIRECT)
    close_mock.assert_called_once_with(91)


def test_cpu_raw_writer_rejects_extent_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A physical map must cover the requested logical wave contiguously."""
    monkeypatch.setattr(os, "O_DIRECT", 0x4000, raising=False)
    with patch("os.open", return_value=92), patch("os.close"):
        writer = CPURawBlockWriter(
            "/dev/snvme0n1",
            target_mib_s=0,
            block_bytes=512,
        )
        with pytest.raises(RuntimeError, match="logical gap"):
            writer.write(
                bytes(1536),
                raw_extents=[
                    (0, 10, 1),
                    (1024, 20, 1),
                ],
                base_file_offset=0,
                logical_nbytes=1536,
            )
        writer.close()


def test_cpu_raw_writer_rejects_write_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the writer makes later writes fail without touching storage."""
    monkeypatch.setattr(os, "O_DIRECT", 0x4000, raising=False)
    monkeypatch.setattr(
        os,
        "pwrite",
        lambda _fd, data, _offset: len(data),
        raising=False,
    )
    with patch("os.open", return_value=93), patch("os.close"):
        writer = CPURawBlockWriter(
            "/dev/snvme0n1",
            target_mib_s=0,
            block_bytes=512,
        )
        writer.close()
        with pytest.raises(RuntimeError, match="closed"):
            writer.write(
                bytes(512),
                raw_extents=[(0, 10, 1)],
                base_file_offset=0,
                logical_nbytes=512,
            )
