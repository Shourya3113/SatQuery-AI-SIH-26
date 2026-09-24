"""
SatQuery AI - Geospatial Dev Workbench Test Suite
Owner: Misha (Database & Geospatial Pipeline Lead)

The workbench is a dev tool, but its mask builders encode two decisions worth
locking down:
  1. an already-calibrated (dB) SAR band must NOT be calibrated a second time;
  2. ``--min-area-ha`` really filters polygons via the projector's own parameter.
"""

import unittest
from pathlib import Path

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from scripts import dev_workbench as wb
except Exception:  # pragma: no cover - exercised on machines without the deps
    np = rasterio = from_origin = wb = None

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
OPT = SAMPLES / "optical.tif"
SAR = SAMPLES / "sar.tif"


def setUpModule():
    if wb is None or np is None or rasterio is None:
        raise unittest.SkipTest("requires rasterio + numpy (project venv)")
    if not OPT.exists() or not SAR.exists():
        raise unittest.SkipTest("sample GeoTIFFs missing — run scripts/make_sample_tiffs.py")


class TestSarCalibrationGuard(unittest.TestCase):
    """The harness must never double-calibrate a band that is already in dB."""

    def test_negative_floor_means_already_db(self):
        self.assertTrue(wb.looks_like_db(np.array([-30.0, -20.0, -0.5], dtype=np.float32)))
        self.assertTrue(wb.looks_like_db(np.linspace(-25.0, -8.0, 64, dtype=np.float32)))

    def test_non_negative_dn_is_not_db(self):
        # Amplitude/DN rasters are non-negative, so they must still be calibrated.
        self.assertFalse(wb.looks_like_db(np.array([0.0, 20.0, 200.0], dtype=np.float32)))
        self.assertFalse(wb.looks_like_db(np.full((8, 8), 80.0, dtype=np.float32)))
        # Bright dB values (above the 25 dB ceiling) are treated as DN, not dB.
        self.assertFalse(wb.looks_like_db(np.array([-10.0, 30.0], dtype=np.float32)))

    def test_sar_sample_is_detected_as_db_and_calibration_skipped(self):
        mask, params = wb.build_sar_mask(str(SAR), regime="surface")
        self.assertTrue(params["already_db"])
        self.assertIn("skipped", params["calibration"])
        # Surface pixels must match the module's own regime classification.
        lo, hi = params["sigma0_range_db"]
        self.assertLess(lo, 0.0)
        self.assertLessEqual(hi, 0.0)
        self.assertEqual(int(mask.sum()), params["regimes"]["counts"]["surface_specular"])
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))

    def test_dn_input_is_calibrated_and_k_cal_changes_the_regime(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dn_sar.tif"
            with rasterio.open(
                p, "w", driver="GTiff", height=16, width=16, count=1,
                dtype="float32", crs="EPSG:32633",
                transform=from_origin(500_000, 5_400_000, 10.0, 10.0),
            ) as ds:
                ds.write(np.full((1, 16, 16), 20.0, dtype=np.float32))
                ds.descriptions = ("VV",)

            # DN=20 -> 10*log10(400) = +26.02 dB with no K_cal: not water.
            _, params = wb.build_sar_mask(str(p), regime="surface", k_cal_db=0.0)
            self.assertFalse(params["already_db"])
            self.assertIn("10*log10", params["calibration"])
            self.assertEqual(params["regimes"]["counts"]["surface_specular"], 0)

            # ...but with the Sentinel-1 GRD K_cal it lands in the calm-water regime.
            mask, params_grd = wb.build_sar_mask(str(p), regime="surface", k_cal_db=83.0)
            self.assertEqual(params_grd["regimes"]["counts"]["surface_specular"], 16 * 16)
            self.assertEqual(int(mask.sum()), 16 * 16)


class TestOpticalMask(unittest.TestCase):
    """Thresholding a real index through the real band resolver."""

    def test_threshold_is_a_strict_upper_bound(self):
        # NDVI on the sample spans about -0.45 .. 0.45, so -0.5 catches every pixel.
        low_mask, low_params = wb.build_optical_mask(str(OPT), "ndvi", threshold=-0.5)
        high_mask, high_params = wb.build_optical_mask(str(OPT), "ndvi", threshold=0.4)
        self.assertGreater(int(low_mask.sum()), int(high_mask.sum()))
        self.assertEqual(int(low_mask.sum()), 128 * 128)
        # index_range is reported so the operator can pick a sane threshold
        self.assertEqual(len(low_params["index_range"]), 2)
        self.assertTrue(set(np.unique(high_mask)).issubset({0, 1}))
        self.assertEqual(high_params["index"], "NDVI")
        self.assertEqual(low_params["band_resolution"]["band_indices"], {"nir": 4, "red": 3})

    def test_swir_index_without_swir_band_raises(self):
        # The sample has no SWIR band, so NDBI must fail loudly rather than fake it.
        with self.assertRaises(ValueError):
            wb.build_optical_mask(str(OPT), "ndbi", threshold=0.0)


class TestRenderOutput(unittest.TestCase):
    """The rendered page must carry the georeferenced overlay and the panel."""

    @unittest.skipUnless(wb is not None and wb._HAVE_FOLIUM, "requires folium")
    def test_payload_renders_standalone_map(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "map.html"
            payload = wb.run(
                raster=str(OPT), mode="optical", index="ndvi", threshold=0.0,
                regime="surface", k_cal=None, min_area_ha=0.0, open_browser=False,
            )
            # run() writes to the repo outputs dir; render again into a temp file.
            wb.render(payload, wb._rgb_preview(wb.gtx.load_raster(str(OPT))), out)
            html = out.read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64", html)   # self-contained raster overlay
            self.assertIn("SatQuery dev workbench", html)  # info panel injected
            self.assertIn("leaflet", html.lower())
            self.assertGreater(payload["mask_pixels"], 0)
            self.assertEqual(payload["geojson"]["type"], "FeatureCollection")

    @unittest.skipUnless(wb is not None and wb._HAVE_FOLIUM, "requires folium")
    def test_min_area_ha_actually_filters_polygons(self):
        raw = wb.run(raster=str(OPT), mode="optical", index="ndvi", threshold=0.2,
                     regime="surface", k_cal=None, min_area_ha=0.0, open_browser=False)
        big = wb.run(raster=str(OPT), mode="optical", index="ndvi", threshold=0.2,
                     regime="surface", k_cal=None, min_area_ha=0.05, open_browser=False)
        self.assertLess(big["geojson"]["satquery"]["n_features"],
                        raw["geojson"]["satquery"]["n_features"])
        self.assertLess(big["geojson"]["satquery"]["total_area_ha"],
                        raw["geojson"]["satquery"]["total_area_ha"])
        for feat in big["geojson"]["features"]:
            self.assertGreaterEqual(feat["properties"]["area_ha"], 0.05)

    @unittest.skipUnless(wb is not None and wb._HAVE_FOLIUM, "requires folium")
    def test_footprint_matches_affine(self):
        payload = wb.run(raster=str(OPT), mode="optical", index="ndvi", threshold=0.2,
                         regime="surface", k_cal=None, min_area_ha=0.0, open_browser=False)
        w_, s_, e_, n_ = payload["raster"]["bounds_wgs84"]
        # 1280 m east of UTM 500000 -> 15.0000 E; spans ~0.0174 deg of longitude.
        self.assertAlmostEqual(w_, 15.0, places=3)
        self.assertAlmostEqual(e_ - w_, 0.0174, places=3)
        self.assertLess(n_ - s_, e_ - w_)  # latitude span is shorter at 48.7 N


if __name__ == "__main__":
    unittest.main()
