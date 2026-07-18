# SPDX-License-Identifier: Apache-2.0
"""Shape-contract tests for the DSv4-Flash one-layer Tutti benchmark."""

# First Party
from benchmarks.tutti_csa_layer_benchmark import (
    DSv4FlashCSALayerGeometry,
    geometry_report,
)


def test_v28_dsv4_flash_csa_layer_geometry() -> None:
    """The V28 benchmark must not regress to the older 30-layer layout."""
    geometry = DSv4FlashCSALayerGeometry()
    geometry.validate()
    report = geometry_report(geometry)

    assert geometry.csa_layers == 21
    assert geometry.num_chunks == 1_874
    assert geometry.cached_tokens == 479_744
    assert report["attention_chunk_shape"] == [64, 1, 584]
    assert report["attention_chunk_bytes"] == 37_376
    assert report["attention_layer_bytes"] == 70_042_624
    assert report["indexer_chunk_shape"] == [64, 1, 132]
    assert report["indexer_chunk_bytes"] == 8_448
    assert report["indexer_layer_bytes"] == 15_831_552
