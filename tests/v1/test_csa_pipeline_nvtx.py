# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.csa_pipeline_nvtx import (
    CsaNvtxEvent,
    CsaNvtxRange,
    CsaPipelineNvtx,
)


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def start_range(self, message: str, color: str, domain: str) -> object:
        range_id = f"range-{len(self.calls)}"
        self.calls.append(("start", range_id, message, color, domain))
        return range_id

    def end_range(self, range_id: object) -> None:
        self.calls.append(("end", range_id))

    def mark(self, message: str, color: str, domain: str) -> None:
        self.calls.append(("mark", message, color, domain))


def test_disabled_tracer_is_noop() -> None:
    backend = RecordingBackend()
    tracer = CsaPipelineNvtx(enabled=False, backend=backend)

    with tracer.range(CsaNvtxEvent.HCA_ATTENTION, layer_id=1):
        pass
    tracer.finish(None)

    assert backend.calls == []


def test_detailed_tracer_uses_independent_environment(
    monkeypatch,
) -> None:
    backend = RecordingBackend()
    monkeypatch.setenv("LMCACHE_CSA_PIPELINE_NVTX", "0")
    monkeypatch.setenv("LMCACHE_CSA_DETAILED_IO_NVTX", "1")

    tracer = CsaPipelineNvtx(
        backend=backend,
        env_name="LMCACHE_CSA_DETAILED_IO_NVTX",
    )
    with tracer.range(
        CsaNvtxEvent.IO_LOADER_CALL,
        layer_id=4,
        target_layer_id=6,
        operation_id="op-1",
        attributes={"kind": "csa_predicted"},
    ):
        pass

    assert tracer.enabled is True
    assert "event=io_loader_call|layer=4|target=6|op=op-1" in str(
        backend.calls[0][2]
    )


def test_sync_range_emits_machine_readable_label() -> None:
    backend = RecordingBackend()
    tracer = CsaPipelineNvtx(enabled=True, backend=backend)

    with tracer.range(
        CsaNvtxEvent.L2_PROXY,
        layer_id=4,
        target_layer_id=6,
        request_id="req|1",
        attributes={"blocks": 8},
    ):
        pass

    assert backend.calls == [
        (
            "start",
            "range-0",
            "event=l2_proxy|layer=4|target=6|request=req%7C1|blocks=8",
            "orange",
            "lmcache.csa.pipeline",
        ),
        ("end", "range-0"),
    ]


def test_io_range_correlates_submit_and_done() -> None:
    backend = RecordingBackend()
    tracer = CsaPipelineNvtx(enabled=True, backend=backend)

    handle = tracer.start_io(
        layer_id=9,
        target_layer_id=11,
        request_id="r1",
        operation_id=27,
    )
    tracer.finish_io(
        handle,
        layer_id=9,
        target_layer_id=11,
        request_id="r1",
        operation_id=27,
    )

    assert [call[0] for call in backend.calls] == ["mark", "start", "end", "mark"]
    assert "event=io_submit|layer=9|target=11|request=r1|op=27" in backend.calls[0]
    assert "event=io_in_flight|layer=9|target=11|request=r1|op=27" in backend.calls[1]
    assert (
        "event=io_done|layer=9|target=11|request=r1|op=27|status=ok" in backend.calls[3]
    )


def test_range_close_is_idempotent() -> None:
    backend = RecordingBackend()
    handle = CsaNvtxRange("id", backend)

    handle.close()
    handle.close()

    assert backend.calls == [("end", "id")]


def test_scatter_range_keeps_hca_csa_kind_and_layer_mapping() -> None:
    backend = RecordingBackend()
    tracer = CsaPipelineNvtx(enabled=True, backend=backend)

    with tracer.range(
        CsaNvtxEvent.IO_SCATTER,
        layer_id=3,
        target_layer_id=5,
        request_id="r2",
        attributes={"kind": "csa_predicted", "blocks": 17},
    ):
        pass

    assert (
        "event=io_scatter|layer=3|target=5|request=r2|blocks=17|"
        "kind=csa_predicted"
    ) in backend.calls[0]
