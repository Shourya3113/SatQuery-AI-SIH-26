"""satquery.db — lightweight local raster registry (replaces full PostGIS).

Track 5 (Misha). ``models.py`` defines the raster record schema; ``connection.py``
owns the SQLite connection and spatial queries. Arrays are never stored — model
code holds ``(C, H, W)`` slices in memory and re-loads from the GeoTIFF path;
this registry keeps metadata + provenance so a raster id can be looked up at
any time (and from the execution trace).
"""
from .connection import (
    DEFAULT_DB_PATH,
    RasterCache,
    close_store,
    get_raster_by_id,
    get_store,
    list_raster_records,
    save_raster_record,
)

__all__ = [
    "RasterCache",
    "DEFAULT_DB_PATH",
    "save_raster_record",
    "get_raster_by_id",
    "list_raster_records",
    "get_store",
    "close_store",
]
