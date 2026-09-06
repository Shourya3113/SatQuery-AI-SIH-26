"""
SatQuery AI - Geospatial Pipeline Test Suite
Owner: Misha (Database & Geospatial Pipeline Lead)
"""

import unittest
from pathlib import Path

try:
    import numpy as np
    import rasterio
    from services import geospatial
except Exception:
    np = rasterio = None
    geospatial = None

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
OPT = SAMPLES / "optical.tif"
SAR = SAMPLES / "sar.tif"

# Affine used by the sample rasters
SAMPLE_TRANSFORM = rasterio.transform.from_origin(500_000, 5_400_000, 10.0, 10.0) if rasterio else None


def setUpModule():
    if geospatial is None or np is None or rasterio is None:
        raise unittest.SkipTest("requires rasterio + numpy (project venv)")
    if not OPT.exists() or not SAR.exists():
        raise unittest.SkipTest("sample GeoTIFFs missing — run scripts/make_sample_tiffs.py")


class TestMockContract(unittest.TestCase):
    """Deliverable 1: Chhavi's Day-1 mock tensors."""

    def test_shapes_dtypes_ranges(self):
        optical, sar = geospatial.generate_mock_tensors(64, seed=1)
        self.assertEqual(optical.shape, (4, 64, 64))
        self.assertEqual(optical.dtype.name, "float32")
        self.assertGreaterEqual(optical.min(), 0.0)
        self.assertLessEqual(optical.max(), 1.0)
        self.assertEqual(sar.shape, (1, 64, 64))
        self.assertEqual(sar.dtype.name, "float32")
        self.assertGreaterEqual(sar.min(), -30.0)
        self.assertLessEqual(sar.max(), 0.0)


class TestRasterLoader(unittest.TestCase):
    """Deliverable 2: metadata extraction + normalized/calibrated model inputs."""

    def test_load_optical_sample(self):
        r = geospatial.load_raster(str(OPT))
        self.assertTrue(r.ok)
        self.assertEqual(r.crs, "EPSG:32633")
        self.assertEqual(r.crs_epsg, 32633)
        self.assertEqual((r.gsd_x, r.gsd_y), (10.0, 10.0))
        self.assertEqual(r.arrays.shape, (4, 128, 128))
        self.assertEqual(r.arrays.dtype.name, "float32")
        self.assertEqual(r.band_names, ["B02", "B03", "B04", "B08"])
        self.assertEqual(r.bounds, [500_000.0, 5_398_720.0, 501_280.0, 5_400_000.0])
        self.assertEqual(r.metadata.get("platform"), "Sentinel-2")

    def test_load_missing_file(self):
        r = geospatial.load_raster(str(SAMPLES / "nope.tif"))
        self.assertFalse(r.ok)
        self.assertTrue(any("not found" in p for p in r.problems))

    def test_model_input_optical_normalized(self):
        x, meta = geospatial.raster_model_input(str(OPT), modality="optical")
        self.assertEqual(x.shape, (4, 128, 128))
        self.assertEqual(x.dtype.name, "float32")
        self.assertGreaterEqual(x.min(), 0.0)
        self.assertLessEqual(x.max(), 1.0)
        self.assertTrue(meta.ok)

    def test_model_input_sar_calibrated_dB(self):
        x, meta = geospatial.raster_model_input(str(SAR), modality="sar")
        raw = geospatial.load_raster(str(SAR)).arrays
        # water patch (DN=20) must calibrate below the built-up patch (DN=200)
        water = x[0, 20:35, 20:35].mean()
        built = x[0, 40:90, 50:100].mean()
        self.assertLess(water, built)
        self.assertEqual(x.shape, raw.shape)
        self.assertEqual(x.dtype.name, "float32")


class TestSARAndIndices(unittest.TestCase):
    """Deliverable 4: sigma0 calibration, adaptive Lee filter, NDVI/NDWI."""

    def test_sigma0_formula(self):
        dn = np.array([10.0, 100.0], dtype=np.float32)
        out = geospatial.calibrate_sigma0(dn)
        self.assertAlmostEqual(float(out[0]), 20.0, places=4)  # 10*log10(100)
        self.assertAlmostEqual(float(out[1]), 40.0, places=4)  # 10*log10(10000)

    def test_sigma0_cal_constant_shift(self):
        out = geospatial.calibrate_sigma0(np.array([10.0]), k_cal_db=10.0)
        self.assertAlmostEqual(float(out[0]), 10.0, places=4)

    def test_sigma0_zero_dn_finite(self):
        out = geospatial.calibrate_sigma0(np.array([0.0]))
        self.assertTrue(np.isfinite(out).all())

    def test_lee_preserves_uniform_and_kills_speckle(self):
        const = np.full((64, 64), 7.0, dtype=np.float32)
        self.assertTrue(np.allclose(geospatial.adaptive_lee_filter(const), 7.0, atol=1e-3))
        speck = np.random.default_rng(0).gamma(2.0, 20.0, (64, 64)).astype(np.float32)
        filtered = geospatial.adaptive_lee_filter(speck, window_size=5, enl=1.0)
        self.assertLess(filtered.std(), speck.std())

    def test_lee_3d_bandwise(self):
        arr = np.stack([np.full((32, 32), 3.0), np.full((32, 32), 9.0)]).astype(np.float32)
        out = geospatial.adaptive_lee_filter(arr)
        self.assertEqual(out.shape, (2, 32, 32))
        self.assertTrue(np.allclose(out[0], 3.0, atol=1e-3))
        self.assertTrue(np.allclose(out[1], 9.0, atol=1e-3))

    def test_lee_rejects_even_window(self):
        with self.assertRaises(ValueError):
            geospatial.adaptive_lee_filter(np.zeros((8, 8)), window_size=4)

    def test_indices_formulas(self):
        self.assertAlmostEqual(float(geospatial.ndvi(0.8, 0.2)), 0.6, places=6)
        self.assertAlmostEqual(float(geospatial.ndwi(0.5, 0.5)), 0.0, places=6)
        self.assertEqual(float(geospatial.ndvi(0.0, 0.0)), 0.0)

    def test_raster_index_resolves_bands(self):
        nd, info = geospatial.raster_index(str(OPT), "ndvi")
        self.assertEqual(nd.shape, (128, 128))
        self.assertEqual(info["nir_band"], "B08")
        self.assertEqual(info["red_band"], "B04")
        r = geospatial.load_raster(str(OPT))
        manual = geospatial.ndvi(r.arrays[3], r.arrays[2])
        self.assertTrue(np.allclose(nd, manual, atol=1e-6))

    def test_unknown_index_raises(self):
        with self.assertRaises(ValueError):
            geospatial.raster_index(str(OPT), "nope")


class TestAffineProjector(unittest.TestCase):
    """Deliverable 3: mask (H, W) -> EPSG:4326 GeoJSON with hectare metrics."""

    def test_block_projects_to_25_ha(self):
        m = np.zeros((128, 128), dtype="uint8")
        m[40:90, 50:100] = 1
        gj = geospatial.mask_to_geojson(m, transform=SAMPLE_TRANSFORM, crs="EPSG:32633", label="built-up")
        self.assertEqual(gj["type"], "FeatureCollection")
        self.assertEqual(len(gj["features"]), 1)
        feat = gj["features"][0]
        self.assertEqual(feat["properties"]["label"], "built-up")
        self.assertAlmostEqual(feat["properties"]["area_ha"], 25.0, places=4)
        self.assertAlmostEqual(gj["satquery"]["total_area_ha"], 25.0, places=4)
        lon, lat = feat["geometry"]["coordinates"][0][0]
        self.assertGreater(lon, 15.0)
        self.assertLess(lon, 15.05)
        self.assertGreater(lat, 48.74)
        self.assertLess(lat, 48.77)

    def test_two_regions_summed_and_filtered(self):
        m = np.zeros((128, 128), dtype="uint8")
        m[40:90, 50:100] = 1
        m[10:20, 10:20] = 1
        gj = geospatial.mask_to_geojson(m, transform=SAMPLE_TRANSFORM, crs="EPSG:32633")
        self.assertEqual(len(gj["features"]), 2)
        self.assertAlmostEqual(gj["satquery"]["total_area_ha"], 26.0, places=4)
        small = geospatial.mask_to_geojson(m, transform=SAMPLE_TRANSFORM, crs="EPSG:32633", min_area_ha=2.0)
        self.assertEqual(len(small["features"]), 1)
        self.assertAlmostEqual(small["features"][0]["properties"]["area_ha"], 25.0, places=4)

    def test_raster_mask_to_geojson_pulls_transform(self):
        m = np.zeros((128, 128), dtype="uint8")
        m[40:90, 50:100] = 1
        gj = geospatial.raster_mask_to_geojson(m, str(OPT))
        self.assertEqual(gj["satquery"]["source_crs"], "EPSG:32633")
        self.assertAlmostEqual(gj["satquery"]["total_area_ha"], 25.0, places=4)

    def test_half_pixel_pixel_to_world(self):
        x, y = geospatial.pixel_to_world(SAMPLE_TRANSFORM, 40, 50)
        self.assertAlmostEqual(x, 500_505.0, places=6)
        self.assertAlmostEqual(y, 5_399_595.0, places=6)
        row, col = geospatial.world_to_pixel(SAMPLE_TRANSFORM, 500_505.0, 5_399_595.0)
        self.assertAlmostEqual(row, 40.0, places=6)
        self.assertAlmostEqual(col, 50.0, places=6)

    def test_geographic_crs_measured_via_utm(self):
        m = np.zeros((10, 10), dtype="uint8")
        m[0, 0] = 1
        tr = rasterio.transform.from_origin(15.0, 48.753, 1e-4, 1e-4)
        gj = geospatial.mask_to_geojson(m, transform=tr, crs="EPSG:4326")
        ha = gj["features"][0]["properties"]["area_ha"]
        self.assertAlmostEqual(ha, 0.0081, delta=0.001)

    def test_requires_2d_mask_and_transform(self):
        with self.assertRaises(ValueError):
            geospatial.mask_to_geojson(np.zeros((2, 8, 8), dtype="uint8"), transform=SAMPLE_TRANSFORM)
        with self.assertRaises(ValueError):
            geospatial.mask_to_geojson(np.zeros((8, 8), dtype="uint8"), transform=None)

    def test_no_shapely_fallback(self):
        orig = geospatial._HAVE_SHAPELY
        geospatial._HAVE_SHAPELY = False
        try:
            m = np.zeros((64, 64), dtype="uint8")
            m[10:20, 10:20] = 1
            gj = geospatial.mask_to_geojson(m, transform=SAMPLE_TRANSFORM, crs="EPSG:32633")
            self.assertEqual(len(gj["features"]), 1)
            self.assertIn("simplify skipped", gj["satquery"]["note"])
        finally:
            geospatial._HAVE_SHAPELY = orig


if __name__ == "__main__":
    unittest.main()
