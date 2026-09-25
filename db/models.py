"""
SatQuery AI - Spatial Database Models & SQLite Schema
Owner: Misha (Database & Geospatial Pipeline Lead)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Dict, Any, Optional
from datetime import datetime


class RasterMetadataRecord:
    """
    Metadata representation of ingested satellite scenes in PostGIS / SQLite.
    """
    def __init__(
        self,
        scene_id: str,
        filename: str,
        sensor: str,  # Cartosat-2S, RISAT-1A, Sentinel-2
        crs: str,
        bounds: list,
        spatial_resolution_m: float
    ):
        self.scene_id = scene_id
        self.filename = filename
        self.sensor = sensor
        self.crs = crs
        self.bounds = bounds
        self.spatial_resolution_m = spatial_resolution_m
        self.created_at = datetime.utcnow()


# Lightweight SQLite Schema for raster metadata & spatial cache (replaces heavy PostGIS)
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rasters (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    path        TEXT NOT NULL,
    modality    TEXT,
    crs         TEXT,
    crs_epsg    INTEGER,
    gsd_x       REAL,
    gsd_y       REAL,
    width       INTEGER,
    height      INTEGER,
    bands       TEXT,        -- JSON array of band names
    bounds      TEXT,        -- JSON [left, bottom, right, top]
    metadata    TEXT,        -- JSON: acquisition tags / trace extras
    problems    TEXT         -- JSON list, kept for the trace
);
"""

# Plain scalar columns (order matches SCHEMA_SQL)
SCALAR_KEYS = ("path", "modality", "crs", "crs_epsg", "gsd_x", "gsd_y", "width", "height")

# Columns stored as JSON text
JSON_KEYS = ("bands", "bounds", "metadata", "problems")

# Empty-container default per JSON column when the record omits it
_JSON_DEFAULTS = {"bands": [], "bounds": None, "metadata": {}, "problems": []}


def record_values(record: dict) -> tuple:
    """Flatten a raster record dict into the 12 non-id column values in
    SCHEMA_SQL order (8 scalars + 4 JSON blobs)."""
    scalars = tuple(record.get(k) for k in SCALAR_KEYS)
    blobs = tuple(json.dumps(record.get(k) if record.get(k) is not None else _JSON_DEFAULTS[k])
                  for k in JSON_KEYS)
    return scalars + blobs


def row_to_record(row: sqlite3.Row) -> dict:
    """Convert a sqlite row back into a plain dict with JSON columns parsed."""
    rec = dict(row)
    for key in JSON_KEYS:
        raw = rec.get(key)
        if raw:
            try:
                rec[key] = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                rec[key] = None
    return rec
