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


class TestSpectralIndicesExtended(unittest.TestCase):
    """MNDWI / NDBI and the SWIR band-resolution contract."""

    def test_mndwi_formula(self):
        # (Green - SWIR) / (Green + SWIR) = (0.4 - 0.1) / 0.5
        self.assertAlmostEqual(float(geospatial.mndwi(0.4, 0.1)), 0.6, places=5)

    def test_ndbi_formula(self):
        # (SWIR - NIR) / (SWIR + NIR) = (0.35 - 0.15) / 0.5
        self.assertAlmostEqual(float(geospatial.ndbi(0.35, 0.15)), 0.4, places=5)

    def test_swir_indices_zero_safe(self):
        self.assertEqual(float(geospatial.ndbi(0.0, 0.0)), 0.0)
        self.assertEqual(float(geospatial.mndwi(0.0, 0.0)), 0.0)

    def test_ndbi_red_proxy_is_documented_as_a_proxy(self):
        # Same numbers through the two paths must NOT be conflated: the proxy is
        # (Red - NIR)/(Red + NIR), real NDBI is (SWIR - NIR)/(SWIR + NIR).
        red, nir, swir = 0.30, 0.10, 0.45
        proxy = float(geospatial.ndbi_red_proxy(red, nir))
        real = float(geospatial.ndbi(swir, nir))
        self.assertAlmostEqual(proxy, 0.5, places=6)
        self.assertAlmostEqual(real, 0.63636363, places=6)
        self.assertNotAlmostEqual(proxy, real, places=3)

    def test_swir_indices_require_a_swir_band(self):
        # The 4-band sample carries no SWIR; it must raise, never silently fake NDBI.
        for name in ("mndwi", "ndbi"):
            with self.assertRaises(ValueError):
                geospatial.raster_index(str(OPT), name)

    def test_raster_index_unknown_name_lists_supported(self):
        with self.assertRaises(ValueError):
            geospatial.raster_index(str(OPT), "nope")

    def test_raster_index_prefers_swir1_over_swir2(self):
        import tempfile

        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ms_swir.tif"
            # Deliberately list B12 BEFORE B11 so priority (not file order) decides.
            data = np.stack([
                np.full((16, 16), 0.10),   # B02 blue
                np.full((16, 16), 0.40),   # B03 green
                np.full((16, 16), 0.20),   # B04 red
                np.full((16, 16), 0.15),   # B08 nir
                np.full((16, 16), 0.90),   # B12 swir2
                np.full((16, 16), 0.35),   # B11 swir1
            ]).astype("float32")
            with rasterio.open(
                p, "w", driver="GTiff", height=16, width=16, count=6,
                dtype="float32", crs="EPSG:32633",
                transform=from_origin(500_000, 5_400_000, 10.0, 10.0),
            ) as dst:
                dst.write(data)
                dst.descriptions = ["B02", "B03", "B04", "B08", "B12", "B11"]

            ndbi_arr, info = geospatial.raster_index(str(p), "ndbi")
            self.assertEqual(info["swir_band"], "B11")
            self.assertEqual(info["nir_band"], "B08")
            self.assertEqual(info["band_indices"], {"swir": 6, "nir": 4})
            # (0.35 - 0.15) / (0.35 + 0.15)
            self.assertAlmostEqual(float(ndbi_arr[0, 0]), 0.4, places=4)

            mndwi_arr, m_info = geospatial.raster_index(str(p), "mndwi")
            self.assertEqual(m_info["green_band"], "B03")
            self.assertEqual(m_info["swir_band"], "B11")
            # (0.40 - 0.35) / (0.40 + 0.35)
            self.assertAlmostEqual(float(mndwi_arr[0, 0]), 0.0666666, places=4)

    def test_unlabelled_raster_uses_positional_fallback(self):
        """Real-world GeoTIFFs often ship empty band descriptions (e.g. rasterio
        window writes); a 4-band B,G,R,NIR stack must still resolve by position."""
        import tempfile

        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "unlabelled.tif"
            data = np.stack([
                np.full((16, 16), 0.10),   # 1: blue
                np.full((16, 16), 0.40),   # 2: green
                np.full((16, 16), 0.20),   # 3: red
                np.full((16, 16), 0.60),   # 4: nir
            ]).astype("float32")
            with rasterio.open(
                p, "w", driver="GTiff", height=16, width=16, count=4,
                dtype="float32", crs="EPSG:32633",
                transform=from_origin(500_000, 5_400_000, 10.0, 10.0),
            ) as dst:
                dst.write(data)  # no descriptions set on purpose

            nd, info = geospatial.raster_index(str(p), "ndvi")
            self.assertEqual(info["band_indices"], {"nir": 4, "red": 3})
            # (0.60 - 0.20) / (0.60 + 0.20)
            self.assertAlmostEqual(float(nd[0, 0]), 0.5, places=4)

            # SWIR indices must still refuse: 4 bands carry no SWIR, positional
            # fallback must not silently invent one.
            with self.assertRaises(ValueError):
                geospatial.raster_index(str(p), "ndbi")

    def test_labelled_raster_without_matching_hints_still_raises(self):
        """The positional fallback must apply ONLY to unlabelled rasters; a
        labelled stack whose names match no hint keeps failing loudly."""
        import tempfile

        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "oddly_labelled.tif"
            data = np.stack([np.full((16, 16), v) for v in (0.1, 0.4, 0.2, 0.6)]).astype("float32")
            with rasterio.open(
                p, "w", driver="GTiff", height=16, width=16, count=4,
                dtype="float32", crs="EPSG:32633",
                transform=from_origin(500_000, 5_400_000, 10.0, 10.0),
            ) as dst:
                dst.write(data)
                dst.descriptions = ["ch1", "ch2", "ch3", "ch4"]

            with self.assertRaises(ValueError):
                geospatial.raster_index(str(p), "ndvi")


class TestSentinel1Calibration(unittest.TestCase):
    """sigma0 (dB) calibration constants and the -16 dB scattering regimes."""

    def test_grd_kcal_constant(self):
        self.assertAlmostEqual(geospatial.SENTINEL1_GRD_K_CAL_DB, 83.0, places=6)
        self.assertAlmostEqual(geospatial.SIGMA0_WATER_MAX_DB, -16.0, places=6)
        self.assertAlmostEqual(geospatial.SIGMA0_VOLUME_MAX_DB, -6.0, places=6)

    def test_default_kcal_unchanged_for_normalized_rasters(self):
        # Backwards contract: the default path stays K_cal = 0.0
        self.assertAlmostEqual(float(geospatial.calibrate_sigma0(np.array([10.0]))[0]), 20.0, places=4)

    def test_sentinel1_helper_applies_grd_kcal(self):
        # DN=100 -> 10*log10(1e4) = 40 dB, then -83.0 dB of K_cal
        out = geospatial.calibrate_sigma0_sentinel1_grd(np.array([100.0], dtype=np.float32))
        self.assertAlmostEqual(float(out[0]), 40.0 - 83.0, places=3)

    def test_calm_water_threshold_is_strictly_below_16_db(self):
        db = np.array([-20.0, -16.0, -15.99, 0.0], dtype=np.float32)
        self.assertEqual(geospatial.calm_water_mask(db).tolist(), [True, False, False, False])

    def test_scattering_regime_labels(self):
        db = np.array([-25.0, -16.0, -10.0, -6.0, -2.0], dtype=np.float32)
        self.assertEqual(geospatial.scattering_regime(db).tolist(), [0, 1, 1, 1, 2])

    def test_classify_regimes_counts_fractions_and_thresholds(self):
        db = np.array([-20.0, -20.0, -10.0, 0.0], dtype=np.float32)
        res = geospatial.classify_sigma0_regimes(db)
        self.assertEqual(res["total_valid_pixels"], 4)
        self.assertEqual(res["counts"]["surface_specular"], 2)
        self.assertEqual(res["counts"]["volume_diffuse"], 1)
        self.assertEqual(res["counts"]["double_bounce"], 1)
        self.assertAlmostEqual(res["fractions"]["surface_specular"], 0.5, places=6)
        self.assertEqual(res["calm_water_threshold_db"], -16.0)
        self.assertEqual(res["thresholds_db"]["double_bounce"], "> -6.0 dB")

    def test_regimes_exclude_non_finite_pixels(self):
        db = np.array([np.nan, np.inf, -20.0], dtype=np.float32)
        res = geospatial.classify_sigma0_regimes(db)
        self.assertEqual(res["total_valid_pixels"], 1)
        self.assertEqual(res["counts"]["surface_specular"], 1)


class TestWindowedCOGReads(unittest.TestCase):
    """Windowed/COG block reading must be byte-identical to the full read."""

    def test_windowed_matches_full_read(self):
        full = geospatial.load_raster(str(OPT))
        win = geospatial.load_raster_windowed(str(OPT), window_size=32)
        self.assertTrue(win.ok and win.real)
        self.assertEqual(win.arrays.shape, full.arrays.shape)
        self.assertEqual(win.arrays.dtype.name, "float32")
        self.assertTrue(np.array_equal(win.arrays, full.arrays))
        self.assertEqual(win.crs, full.crs)
        self.assertEqual(win.crs_epsg, full.crs_epsg)
        self.assertEqual(win.bounds, full.bounds)
        self.assertEqual((win.gsd_x, win.gsd_y), (full.gsd_x, full.gsd_y))
        self.assertEqual(win.band_names, full.band_names)
        self.assertEqual(win.metadata.get("platform"), "Sentinel-2")
        self.assertIn("windowed", win.metadata["satquery_read_mode"])

    def test_iter_windows_tiles_the_raster_exactly_once(self):
        covered = 0
        blocks = 0
        for window, block in geospatial.iter_windows(str(OPT), window_size=48):
            self.assertEqual(block.shape[0], 4)
            self.assertEqual(block.shape[1], int(window.height))
            self.assertEqual(block.shape[2], int(window.width))
            covered += int(window.width) * int(window.height)
            blocks += 1
        # ceil(128 / 48) = 3 blocks per axis, no overlap, no gaps
        self.assertEqual(blocks, 9)
        self.assertEqual(covered, 128 * 128)

    def test_windowed_band_subset(self):
        win = geospatial.load_raster_windowed(str(OPT), window_size=64, bands=[4])
        self.assertEqual(win.arrays.shape, (1, 128, 128))
        self.assertEqual(win.band_names, ["B08"])

    def test_out_of_range_band_rejected(self):
        with self.assertRaises(ValueError):
            geospatial.load_raster_windowed(str(OPT), bands=[9])

    def test_bad_window_size_rejected(self):
        with self.assertRaises(ValueError):
            list(geospatial.iter_windows(str(OPT), window_size=0))

    def test_windowed_missing_file(self):
        r = geospatial.load_raster_windowed(str(SAMPLES / "nope.tif"))
        self.assertFalse(r.ok)
        self.assertTrue(any("not found" in p for p in r.problems))

    def test_streaming_stats_match_numpy(self):
        stats = geospatial.raster_stats_windowed(str(OPT), window_size=32)
        full = geospatial.load_raster(str(OPT))
        self.assertEqual(stats["n_blocks"], 16)
        self.assertEqual(len(stats["bands"]), 4)
        for i, band in enumerate(stats["bands"]):
            self.assertEqual(band["band"], full.band_names[i])
            self.assertEqual(band["index"], i + 1)
            self.assertEqual(band["count"], 128 * 128)
            self.assertAlmostEqual(band["min"], float(full.arrays[i].min()), places=5)
            self.assertAlmostEqual(band["max"], float(full.arrays[i].max()), places=5)
            self.assertAlmostEqual(band["mean"], float(full.arrays[i].mean()), places=5)
            self.assertAlmostEqual(band["std"], float(full.arrays[i].std()), places=5)


class TestEngineContracts(unittest.TestCase):
    """GeospatialEngine wrappers keep their orchestrator-facing behaviour."""

    def test_indices_skip_swir_indices_without_swir(self):
        bands = np.zeros((4, 8, 8), dtype=np.float32)
        indices = geospatial.GeospatialEngine.calculate_spectral_indices(bands)
        self.assertIn("NDVI", indices)
        self.assertIn("NDWI", indices)
        # No SWIR band -> NDBI/MNDWI must be absent, not faked from red
        self.assertNotIn("NDBI", indices)
        self.assertNotIn("MNDWI", indices)

    def test_indices_include_swir_indices_when_present(self):
        bands = np.zeros((5, 8, 8), dtype=np.float32)
        bands[1] = 0.45  # green
        bands[3] = 0.10  # nir
        bands[4] = 0.15  # swir1
        indices = geospatial.GeospatialEngine.calculate_spectral_indices(bands)
        self.assertIn("NDBI", indices)
        self.assertIn("MNDWI", indices)
        # NDBI  = (0.15 - 0.10) / 0.25 = 0.2
        # MNDWI = (0.45 - 0.15) / 0.60 = 0.5
        self.assertAlmostEqual(float(indices["NDBI"][0, 0]), 0.2, places=5)
        self.assertAlmostEqual(float(indices["MNDWI"][0, 0]), 0.5, places=5)

    def test_sar_kcal_defaults_to_normalized_behaviour(self):
        const = np.full((1, 16, 16), 100.0, dtype=np.float32)
        out = geospatial.GeospatialEngine.calibrate_and_filter_sar(const)
        # constant input -> Lee filter is the identity, so sigma0 survives exactly
        self.assertTrue(np.allclose(out, 40.0, atol=1e-3))

    def test_sar_kcal_can_be_set_for_sentinel1_grd(self):
        const = np.full((1, 16, 16), 100.0, dtype=np.float32)
        out = geospatial.GeospatialEngine.calibrate_and_filter_sar(
            const, k_cal_db=geospatial.SENTINEL1_GRD_K_CAL_DB
        )
        self.assertTrue(np.allclose(out, 40.0 - geospatial.SENTINEL1_GRD_K_CAL_DB, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
