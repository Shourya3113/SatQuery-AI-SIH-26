"""
Database/cache connection layer for the raster registry (Track 5).
Owner: Misha (Database & Geospatial Pipeline Lead)

A single-file SQLite store — no Docker PostGIS, no connection flakiness.
Schema lives in db.models; this module owns the connection and spatial queries.
"""
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
import uuid
from pathlib import Path

from .models import SCHEMA_SQL, record_values, row_to_record

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "satquery_cache.db"


class RasterCache:
    """SQLite-backed spatial registry.

    db_path may be a file path or ":memory:"; defaults to the repo's
    data/cache/satquery_cache.db (override with SATQUERY_CACHE_DB).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = os.environ.get("SATQUERY_CACHE_DB", DEFAULT_DB_PATH)
        self._memory = str(db_path) == ":memory:"
        path = Path(":memory:") if self._memory else Path(db_path)
        if not self._memory:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA_SQL)
        self._conn.commit()

    def save(self, record: dict) -> str:
        """Insert/overwrite one raster record; returns its raster_id."""
        rid = str(record.get("id") or uuid.uuid4().hex)
        now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT INTO rasters (id, created_at, path, modality, crs, crs_epsg,"
            " gsd_x, gsd_y, width, height, bands, bounds, metadata, problems)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            " created_at=excluded.created_at, path=excluded.path,"
            " modality=excluded.modality, crs=excluded.crs,"
            " crs_epsg=excluded.crs_epsg, gsd_x=excluded.gsd_x,"
            " gsd_y=excluded.gsd_y, width=excluded.width, height=excluded.height,"
            " bands=excluded.bands, bounds=excluded.bounds,"
            " metadata=excluded.metadata, problems=excluded.problems",
            (rid, now, *record_values(record)),
        )
        self._conn.commit()
        return rid

    def get(self, raster_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM rasters WHERE id = ?", (raster_id,)).fetchone()
        return row_to_record(row) if row else None

    def all(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM rasters ORDER BY created_at DESC").fetchall()
        return [row_to_record(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


# --- module-level convenience API (default store) ---------------------------
_default: RasterCache | None = None


def get_store() -> RasterCache:
    """Lazily opened default store (respects SATQUERY_CACHE_DB)."""
    global _default
    if _default is None:
        _default = RasterCache()
    return _default


def close_store() -> None:
    global _default
    if _default is not None:
        try:
            _default.close()
        except Exception:
            pass
        _default = None


def save_raster_record(metadata: dict) -> str:
    """Persist a raster metadata dict and return the generated raster_id."""
    return get_store().save(metadata)


def get_raster_by_id(raster_id: str) -> dict | None:
    """Fetch a saved raster record by id, or None."""
    return get_store().get(raster_id)


def list_raster_records() -> list[dict]:
    """All saved records, newest first."""
    return get_store().all()
