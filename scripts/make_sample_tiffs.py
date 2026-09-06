"""
Generate sample synthetic GeoTIFFs for SatQuery AI tests and benchmarks.
Creates data/samples/optical.tif and data/samples/sar.tif in EPSG:32633.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

OPTICAL_PATH = SAMPLE_DIR / "optical.tif"
SAR_PATH = SAMPLE_DIR / "sar.tif"

WIDTH = 128
HEIGHT = 128
TRANSFORM = from_origin(500_000, 5_400_000, 10.0, 10.0)
CRS = "EPSG:32633"


def create_samples():
    # 1. Optical 4-band raster: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
    optical_data = np.full((4, HEIGHT, WIDTH), 50.0, dtype=np.float32)
    # Background vegetation: NIR > Red
    optical_data[3] = 120.0  # NIR
    optical_data[2] = 40.0   # Red
    optical_data[1] = 60.0   # Green
    optical_data[0] = 30.0   # Blue

    # Water patch (20:35, 20:35): Low reflectance across bands, Green > NIR
    optical_data[:, 20:35, 20:35] = 20.0
    optical_data[1, 20:35, 20:35] = 25.0  # Green
    optical_data[3, 20:35, 20:35] = 10.0  # NIR

    # Built-up patch (40:90, 50:100): High reflectance
    optical_data[:, 40:90, 50:100] = 200.0

    with rasterio.open(
        OPTICAL_PATH,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=4,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
    ) as ds:
        ds.write(optical_data)
        ds.descriptions = ("B02", "B03", "B04", "B08")
        ds.update_tags(platform="Sentinel-2", acquisition_date="2023-06-15")

    print(f"Created {OPTICAL_PATH}")

    # 2. SAR single-band raster (VV backscatter)
    sar_data = np.full((1, HEIGHT, WIDTH), 80.0, dtype=np.float32)
    # Water patch: specular reflection, low DN = 20
    sar_data[0, 20:35, 20:35] = 20.0
    # Built-up patch: double-bounce corner reflection, high DN = 200
    sar_data[0, 40:90, 50:100] = 200.0

    with rasterio.open(
        SAR_PATH,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
    ) as ds:
        ds.write(sar_data)
        ds.descriptions = ("VV",)
        ds.update_tags(platform="Sentinel-1", acquisition_date="2023-06-15")

    print(f"Created {SAR_PATH}")


if __name__ == "__main__":
    create_samples()
