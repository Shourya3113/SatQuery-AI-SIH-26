import numpy as np
import pytest

from tools.change_engine import BiTemporalChangeEngine


@pytest.fixture(scope="module")
def engine():
    return BiTemporalChangeEngine()


def make_pair(
    bands: int,
    height: int = 64,
    width: int = 80,
):
    rng = np.random.default_rng(42)

    raster_t1 = rng.random(
        (bands, height, width),
        dtype=np.float32,
    )

    raster_t2 = raster_t1.copy()

    # Add a deterministic changed region.
    raster_t2[
        :,
        height // 4:height // 2,
        width // 4:width // 2,
    ] += 0.25

    return raster_t1, raster_t2


@pytest.mark.parametrize("bands", [1, 3, 4])
def test_bit_supports_multiple_modalities(engine, bands):
    raster_t1, raster_t2 = make_pair(bands)

    output, telemetry = engine.execute(
        {
            "raster_t1": raster_t1,
            "raster_t2": raster_t2,
        },
        {
            "backend": "bit",
        },
    )

    assert output["binary_mask"].shape == (64, 80)
    assert output["binary_mask"].dtype == np.uint8
    assert 0.0 <= output["change_percentage"] <= 100.0
    assert telemetry["model"] == "BiT"
    assert telemetry["backend"] == "pretrained_bit"


def test_bit_handles_different_sizes_and_invalid_values(engine):
    rng = np.random.default_rng(123)

    raster_t1 = rng.random(
        (3, 72, 96),
        dtype=np.float32,
    )

    raster_t2 = rng.random(
        (3, 80, 64),
        dtype=np.float32,
    )

    raster_t1[0, 0, 0] = np.nan
    raster_t2[1, 0, 0] = np.inf

    output, telemetry = engine.execute(
        {
            "raster_t1": raster_t1,
            "raster_t2": raster_t2,
        },
        {
            "backend": "bit",
        },
    )

    assert output["binary_mask"].shape == (72, 96)
    assert output["binary_mask"].dtype == np.uint8
    assert 0.0 <= output["change_percentage"] <= 100.0
    assert telemetry["backend"] == "pretrained_bit"


def test_fallback_zero_change(engine):
    raster = np.ones(
        (1, 64, 64),
        dtype=np.float32,
    )

    output, telemetry = engine.execute(
        {
            "raster_t1": raster,
            "raster_t2": raster.copy(),
        },
        {
            "backend": "fallback",
            "change_threshold": 0.20,
        },
    )

    assert output["binary_mask"].shape == (64, 64)
    assert int(output["binary_mask"].sum()) == 0
    assert output["change_percentage"] == 0.0
    assert telemetry["model"] == "DETERMINISTIC-FALLBACK"