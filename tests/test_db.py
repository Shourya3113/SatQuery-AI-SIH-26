"""
SatQuery AI - Spatial Registry / SQLite Cache Test Suite
Owner: Misha (Database & Geospatial Pipeline Lead)
"""

import os
import tempfile
import unittest
from pathlib import Path

import db


def _sample_record(path="optical.tif", modality="optical", rid=None):
    return {
        "id": rid,
        "path": path,
        "modality": modality,
        "crs": "EPSG:32633",
        "crs_epsg": 32633,
        "gsd_x": 10.0,
        "gsd_y": 10.0,
        "width": 128,
        "height": 128,
        "bands": ["B02", "B03", "B04", "B08"],
        "bounds": [500_000.0, 5_398_720.0, 501_280.0, 5_400_000.0],
        "metadata": {"platform": "Sentinel-2", "acquisition_date": "2023-06-15"},
    }


class TestRasterCache(unittest.TestCase):
    """Deliverable 5: lightweight SQLite registry."""

    def setUp(self):
        self.store = db.RasterCache(":memory:")

    def tearDown(self):
        self.store.close()

    def test_save_and_get_roundtrip(self):
        rid = self.store.save(_sample_record())
        got = self.store.get(rid)
        self.assertIsNotNone(got)
        self.assertEqual(got["crs"], "EPSG:32633")
        self.assertEqual(got["bands"], ["B02", "B03", "B04", "B08"])
        self.assertEqual(got["bounds"], [500_000.0, 5_398_720.0, 501_280.0, 5_400_000.0])
        self.assertEqual(got["metadata"]["platform"], "Sentinel-2")
        self.assertTrue(got["id"])

    def test_missing_id_returns_none(self):
        self.assertIsNone(self.store.get("does-not-exist"))

    def test_upsert_by_id(self):
        rid = self.store.save(_sample_record(rid="abc123"))
        self.store.save(_sample_record(rid="abc123", path="other.tif"))
        got = self.store.get("abc123")
        self.assertEqual(rid, "abc123")
        self.assertEqual(got["path"], "other.tif")
        self.assertEqual(len(self.store.all()), 1)

    def test_all_lists_records(self):
        self.store.save(_sample_record(path="a.tif"))
        self.store.save(_sample_record(path="b.tif", modality="sar"))
        recs = self.store.all()
        self.assertEqual(len(recs), 2)
        self.assertEqual({r["path"] for r in recs}, {"a.tif", "b.tif"})

    def test_integration_with_geospatial_metadata(self):
        from services.geospatial import load_raster

        meta = load_raster(str(Path(__file__).resolve().parent.parent / "data" / "samples" / "optical.tif")).to_dict()
        meta["modality"] = "optical"
        rid = self.store.save(meta)
        got = self.store.get(rid)
        self.assertEqual(got["crs_epsg"], 32633)
        self.assertEqual(got["width"], 128)
        self.assertEqual(len(got["bands"]), 4)


class TestModuleFunctions(unittest.TestCase):
    """save_raster_record / get_raster_by_id contract against a temp file."""

    def test_default_store_functions(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "cache.db")
            old = os.environ.get("SATQUERY_CACHE_DB")
            os.environ["SATQUERY_CACHE_DB"] = db_path
            try:
                db.close_store()
                rid = db.save_raster_record(_sample_record())
                got = db.get_raster_by_id(rid)
                self.assertEqual(got["crs"], "EPSG:32633")
                self.assertIsNone(db.get_raster_by_id("nope"))
                self.assertEqual(len(db.list_raster_records()), 1)
            finally:
                db.close_store()
                if old is None:
                    os.environ.pop("SATQUERY_CACHE_DB", None)
                else:
                    os.environ["SATQUERY_CACHE_DB"] = old


if __name__ == "__main__":
    unittest.main()
