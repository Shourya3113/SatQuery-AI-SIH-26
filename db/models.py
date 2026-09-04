"""
SatQuery AI - PostGIS Spatial Database Models
Owner: Misha (Database & Geospatial Pipeline Lead)
"""

from typing import Dict, Any, Optional
from datetime import datetime


class RasterMetadataRecord:
    """
    Metadata representation of ingested satellite scenes in PostGIS.
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
