"""Raster record definitions & table schema for the spatial cache (Track 5).

Pure schema layer — no connection logic. ``connection.py`` executes these
definitions against a SQLite file. Keeping schema + row (de)serialization here
mirrors the ``satquery.db.models`` contract from the architecture plan.
"""
from __future__ import annotations

import json
import sqlite3

# One table is enough for the hackathon: uploaded rasters, bounds, band layout
# and provenance, keyed by a generated raster id. Arrays are never stored —
# model code re-loads ``(C, H, W)`` slices from the GeoTIFF path.
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

# Plain scalar columns (order matches SCHEMA_SQL).
SCALAR_KEYS = ("path", "modality", "crs", "crs_epsg", "gsd_x", "gsd_y", "width", "height")

# Columns stored as JSON text.
JSON_KEYS = ("bands", "bounds", "metadata", "problems")

# Empty-container default per JSON column when the record omits it.
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
