"""
SatQuery AI - CRS & Affine Projection Contract Test Suite
Owner: Misha (Database & Geospatial Pipeline Lead)

Guards the "zero coordinate hallucination" rule: every coordinate the vectorization
pipeline emits must be derived from the raster's 6-parameter affine transform, and
every area must be measured in metres on the ellipsoid BEFORE any degree reprojection.

Covers:
  - EPSG:32633 (UTM) source        -> GeoJSON in EPSG:4326 (default) with exact hectares
  - EPSG:4326 (geographic) source  -> area measured via auto-UTM, never in degrees^2
  - Delhi footprint                -> resolves to UTM 43N (EPSG:32643) per the ISRO spec
  - explicit target_crs override   -> both source and target CRS recorded in the payload
  - pixel <-> world affine round-trips with PixelIsArea half-pixel alignment
"""

import unittest
from pathlib import Path

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin, xy
    from rasterio.warp import transform as rio_transform
    from services import geospatial
except Exception:  # pragma: no cover - exercised on machines without the deps
    np = rasterio = from_origin = xy = rio_transform = None
    geospatial = None

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
OPT = SAMPLES / "optical.tif"
SAR = SAMPLES / "sar.tif"

# Affine of the sample rasters (see scripts/make_sample_tiffs.py):
# origin (500_000, 5_400_000), 10 m pixels, EPSG:32633.
SAMPLE_TRANSFORM = from_origin(500_000, 5_400_000, 10.0, 10.0) if rasterio else None

# The 50x50 px @10 m block used throughout: rows 40:90, cols 50:100.
BLOCK_ROWS = (40, 90)
BLOCK_COLS = (50, 100)


def setUpModule():
    if geospatial is None or np is None or rasterio is None:
        raise unittest.SkipTest("requires rasterio + numpy (project venv)")
    if not OPT.exists() or not SAR.exists():
        raise unittest.SkipTest("sample GeoTIFFs missing — run scripts/make_sample_tiffs.py")


def _block_mask(shape=(128, 128)):
    mask = np.zeros(shape, dtype="uint8")
    mask[BLOCK_ROWS[0]:BLOCK_ROWS[1], BLOCK_COLS[0]:BLOCK_COLS[1]] = 1
    return mask


def _utm_corners():
    """The block's source-CRS corner coordinates, derived from the affine alone."""
    x0 = SAMPLE_TRANSFORM.c + SAMPLE_TRANSFORM.a * BLOCK_COLS[0]
    x1 = SAMPLE_TRANSFORM.c + SAMPLE_TRANSFORM.a * BLOCK_COLS[1]
    y0 = SAMPLE_TRANSFORM.f + SAMPLE_TRANSFORM.e * BLOCK_ROWS[0]  # top (e is negative)
    y1 = SAMPLE_TRANSFORM.f + SAMPLE_TRANSFORM.e * BLOCK_ROWS[1]  # bottom
    return [x0, x1, x0, x1], [y0, y0, y1, y1]


class TestCRSContract(unittest.TestCase):
    """Output CRS standardisation and transform-derived coordinates."""

    def test_projected_source_defaults_to_wgs84_output(self):
        gj = geospatial.mask_to_geojson(
            _block_mask(), transform=SAMPLE_TRANSFORM, crs="EPSG:32633", label="built-up"
        )
        self.assertEqual(gj["satquery"]["source_crs"], "EPSG:32633")
        self.assertEqual(gj["satquery"]["target_crs"], "EPSG:4326")
        self.assertEqual(gj["type"], "FeatureCollection")
        self.assertEqual(len(gj["features"]), 1)

    def test_coordinates_are_affine_derived_not_hardcoded(self):
        """Independent check: reproject the affine's own corners and compare."""
        gj = geospatial.mask_to_geojson(_block_mask(), transform=SAMPLE_TRANSFORM, crs="EPSG:32633")
        ring = gj["features"][0]["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]

        xs, ys = _utm_corners()
        exp_lons, exp_lats = rio_transform("EPSG:32633", "EPSG:4326", xs, ys)

        self.assertAlmostEqual(min(lons), min(exp_lons), places=5)
        self.assertAlmostEqual(max(lons), max(exp_lons), places=5)
        self.assertAlmostEqual(min(lats), min(exp_lats), places=5)
        self.assertAlmostEqual(max(lats), max(exp_lats), places=5)

        # ...and nothing resembling the loose Delhi bbox the stub tools carry around.
        self.assertLess(max(lons), 20.0)
        self.assertGreater(min(lons), 15.0)
        self.assertGreater(min(lats), 48.74)
        self.assertLess(max(lats), 48.77)

    def test_area_is_exact_hectares_measured_before_reprojection(self):
        # 50 x 50 px at 10 m -> 500 m x 500 m -> 250 000 m^2 = 25 ha = 0.25 km^2
        gj = geospatial.mask_to_geojson(_block_mask(), transform=SAMPLE_TRANSFORM, crs="EPSG:32633")
        feat = gj["features"][0]["properties"]
        self.assertAlmostEqual(feat["area_m2"], 250_000.0, places=3)
        self.assertAlmostEqual(feat["area_ha"], 25.0, places=6)
        self.assertAlmostEqual(feat["area_km2"], 0.25, places=8)
        self.assertIn("metres", feat["measurement"])
        self.assertAlmostEqual(gj["satquery"]["total_area_ha"], 25.0, places=6)

    def test_total_area_equals_sum_of_feature_areas(self):
        mask = _block_mask()
        mask[10:20, 10:20] = 1  # 10 x 10 px -> 1 ha
        gj = geospatial.mask_to_geojson(mask, transform=SAMPLE_TRANSFORM, crs="EPSG:32633")
        self.assertEqual(len(gj["features"]), 2)
        summed = sum(f["properties"]["area_ha"] for f in gj["features"])
        self.assertAlmostEqual(summed, gj["satquery"]["total_area_ha"], places=6)
        self.assertAlmostEqual(summed, 26.0, places=6)

    def test_geographic_source_area_measured_in_utm_not_degrees(self):
        # A 1-pixel 1e-4 deg cell near the sample origin; in degrees^2 the shoelace
        # area would be ~1e-8, so a real ellipsoid area proves the auto-UTM path ran.
        mask = np.zeros((10, 10), dtype="uint8")
        mask[0, 0] = 1
        tr = from_origin(15.0, 48.753, 1e-4, 1e-4)
        gj = geospatial.mask_to_geojson(mask, transform=tr, crs="EPSG:4326")
        props = gj["features"][0]["properties"]
        self.assertIn("EPSG:32633", props["measurement"])
        self.assertGreater(props["area_ha"], 0.001)
        self.assertLess(props["area_ha"], 0.02)
        self.assertAlmostEqual(props["area_ha"], 0.0081, delta=0.001)

    def test_delhi_footprint_resolves_to_utm_43n(self):
        # lon 77.10 / lat 28.60 -> UTM zone 43N (EPSG:32643), the ISRO target zone.
        mask = np.zeros((10, 10), dtype="uint8")
        mask[0, 0] = 1
        tr = from_origin(77.10, 28.60, 1e-4, 1e-4)
        gj = geospatial.mask_to_geojson(mask, transform=tr, crs="EPSG:4326")
        props = gj["features"][0]["properties"]
        self.assertIn("EPSG:32643", props["measurement"])
        # ~11 m x ~10 m cell -> ~0.01 ha
        self.assertGreater(props["area_ha"], 0.005)
        self.assertLess(props["area_ha"], 0.02)

    def test_target_crs_override_records_source_and_target(self):
        mask = _block_mask()
        default = geospatial.mask_to_geojson(mask, transform=SAMPLE_TRANSFORM, crs="EPSG:32633")
        override = geospatial.mask_to_geojson(
            mask, transform=SAMPLE_TRANSFORM, crs="EPSG:32633", target_crs="EPSG:32643"
        )
        self.assertEqual(override["satquery"]["source_crs"], "EPSG:32633")
        self.assertEqual(override["satquery"]["target_crs"], "EPSG:32643")
        # Area stays the source-CRS metric value regardless of output projection.
        self.assertAlmostEqual(override["satquery"]["total_area_ha"], 25.0, places=6)
        self.assertAlmostEqual(override["satquery"]["total_area_ha"],
                               default["satquery"]["total_area_ha"], places=6)
        # The emitted geometry really did move to the other CRS.
        self.assertNotAlmostEqual(
            override["features"][0]["geometry"]["coordinates"][0][0][0],
            default["features"][0]["geometry"]["coordinates"][0][0][0],
            places=2,
        )

    def test_raster_mask_to_geojson_pulls_crs_from_file(self):
        gj = geospatial.raster_mask_to_geojson(_block_mask(), str(OPT))
        self.assertEqual(gj["satquery"]["source_crs"], "EPSG:32633")
        self.assertAlmostEqual(gj["satquery"]["total_area_ha"], 25.0, places=6)
        lon = gj["features"][0]["geometry"]["coordinates"][0][0][0]
        self.assertGreater(lon, 15.0)
        self.assertLess(lon, 15.05)

    def test_requires_2d_mask_and_transform(self):
        with self.assertRaises(ValueError):
            geospatial.mask_to_geojson(np.zeros((2, 8, 8), dtype="uint8"), transform=SAMPLE_TRANSFORM)
        with self.assertRaises(ValueError):
            geospatial.mask_to_geojson(np.zeros((8, 8), dtype="uint8"), transform=None)


class TestAffineRoundTrip(unittest.TestCase):
    """pixel <-> world must be exact inverses through the 6-parameter affine."""

    def test_roundtrip_over_a_pixel_grid(self):
        for row in (0, 1, 40, 63, 127):
            for col in (0, 7, 50, 99, 127):
                x, y = geospatial.pixel_to_world(SAMPLE_TRANSFORM, row, col)
                back_row, back_col = geospatial.world_to_pixel(SAMPLE_TRANSFORM, x, y)
                self.assertAlmostEqual(back_row, row, places=6)
                self.assertAlmostEqual(back_col, col, places=6)

    def test_half_pixel_centre_matches_rasterio(self):
        for row, col in ((0, 0), (40, 50), (89, 99), (127, 127)):
            mine = geospatial.pixel_to_world(SAMPLE_TRANSFORM, row, col)
            theirs = xy(SAMPLE_TRANSFORM, row, col, offset="center")
            self.assertAlmostEqual(mine[0], float(theirs[0]), places=6)
            self.assertAlmostEqual(mine[1], float(theirs[1]), places=6)

    def test_singular_transform_rejected(self):
        from affine import Affine

        with self.assertRaises(ValueError):
            geospatial.world_to_pixel(Affine(0, 0, 0, 0, 0, 0), 1.0, 1.0)

    def test_sample_raster_really_uses_the_test_affine(self):
        """If the samples are regenerated at another origin, fail loudly here."""
        with rasterio.open(OPT) as ds:
            self.assertEqual(ds.crs.to_string(), "EPSG:32633")
            self.assertAlmostEqual(ds.transform.a, SAMPLE_TRANSFORM.a, places=9)
            self.assertAlmostEqual(ds.transform.b, SAMPLE_TRANSFORM.b, places=9)
            self.assertAlmostEqual(ds.transform.c, SAMPLE_TRANSFORM.c, places=9)
            self.assertAlmostEqual(ds.transform.d, SAMPLE_TRANSFORM.d, places=9)
            self.assertAlmostEqual(ds.transform.e, SAMPLE_TRANSFORM.e, places=9)
            self.assertAlmostEqual(ds.transform.f, SAMPLE_TRANSFORM.f, places=9)


if __name__ == "__main__":
    unittest.main()
