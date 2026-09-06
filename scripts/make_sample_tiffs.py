"""
Generate sample synthetic GeoTIFFs and PNGs for SatQuery AI tests, demos, and benchmarks.
Creates:
- data/samples/optical.tif (4-band Sentinel-2 RGB+NIR)
- data/samples/sar.tif (1-band Sentinel-1 C-band SAR backscatter)
- data/samples/bitemporal_t1.tif (Pre-event scene)
- data/samples/bitemporal_t2.tif (Post-event flood/expansion scene)
- Corresponding RGB preview PNGs for quick browser testing.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

OPTICAL_PATH = SAMPLE_DIR / "optical.tif"
SAR_PATH = SAMPLE_DIR / "sar.tif"
T1_PATH = SAMPLE_DIR / "bitemporal_t1.tif"
T2_PATH = SAMPLE_DIR / "bitemporal_t2.tif"

WIDTH = 128
HEIGHT = 128
TRANSFORM = from_origin(500_000, 5_400_000, 10.0, 10.0)
CRS = "EPSG:32633"


def save_rgb_png(raster_data: np.ndarray, path: Path):
    """Converts 3-band or 4-band float/uint data to an 8-bit RGB PNG."""
    if raster_data.ndim == 3 and raster_data.shape[0] >= 3:
        # Bands 2, 1, 0 correspond to Red, Green, Blue
        rgb = np.stack([raster_data[2], raster_data[1], raster_data[0]], axis=-1)
    elif raster_data.ndim == 3 and raster_data.shape[0] == 1:
        # Grayscale single-band replicated to RGB
        gray = raster_data[0]
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        rgb = np.stack([raster_data, raster_data, raster_data], axis=-1)

    norm = (rgb - np.min(rgb)) / (np.ptp(rgb) + 1e-6) * 255.0
    img = Image.fromarray(norm.astype(np.uint8))
    img.save(path)
    print(f"Created preview PNG: {path}")


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
        OPTICAL_PATH, "w", driver="GTiff",
        height=HEIGHT, width=WIDTH, count=4,
        dtype="float32", crs=CRS, transform=TRANSFORM,
    ) as ds:
        ds.write(optical_data)
        ds.descriptions = ("B02", "B03", "B04", "B08")
        ds.update_tags(platform="Sentinel-2", acquisition_date="2023-06-15")

    print(f"Created {OPTICAL_PATH}")
    save_rgb_png(optical_data, SAMPLE_DIR / "optical_preview.png")

    # 2. SAR single-band raster (VV backscatter)
    sar_data = np.full((1, HEIGHT, WIDTH), 80.0, dtype=np.float32)
    sar_data[0, 20:35, 20:35] = 20.0
    sar_data[0, 40:90, 50:100] = 200.0

    with rasterio.open(
        SAR_PATH, "w", driver="GTiff",
        height=HEIGHT, width=WIDTH, count=1,
        dtype="float32", crs=CRS, transform=TRANSFORM,
    ) as ds:
        ds.write(sar_data)
        ds.descriptions = ("VV",)
        ds.update_tags(platform="Sentinel-1", acquisition_date="2023-06-15")

    print(f"Created {SAR_PATH}")
    save_rgb_png(sar_data, SAMPLE_DIR / "sar_preview.png")

    # 3. Bi-temporal Pair (T1: baseline, T2: flooded + urban growth)
    t1_data = optical_data.copy()
    with rasterio.open(
        T1_PATH, "w", driver="GTiff",
        height=HEIGHT, width=WIDTH, count=4,
        dtype="float32", crs=CRS, transform=TRANSFORM,
    ) as ds:
        ds.write(t1_data)
        ds.descriptions = ("B02", "B03", "B04", "B08")
        ds.update_tags(platform="Sentinel-2", acquisition_date="2023-01-10")
    print(f"Created {T1_PATH}")
    save_rgb_png(t1_data, SAMPLE_DIR / "bitemporal_t1_preview.png")

    t2_data = optical_data.copy()
    # Expand water area significantly (inundation)
    t2_data[:, 15:55, 15:55] = 20.0
    t2_data[1, 15:55, 15:55] = 25.0
    t2_data[3, 15:55, 15:55] = 10.0
    # Add new construction zone
    t2_data[:, 95:120, 10:40] = 220.0

    with rasterio.open(
        T2_PATH, "w", driver="GTiff",
        height=HEIGHT, width=WIDTH, count=4,
        dtype="float32", crs=CRS, transform=TRANSFORM,
    ) as ds:
        ds.write(t2_data)
        ds.descriptions = ("B02", "B03", "B04", "B08")
        ds.update_tags(platform="Sentinel-2", acquisition_date="2023-08-20")
    print(f"Created {T2_PATH}")
    save_rgb_png(t2_data, SAMPLE_DIR / "bitemporal_t2_preview.png")


if __name__ == "__main__":
    create_samples()
