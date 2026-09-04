"""
SatQuery AI - Geospatial Engine
Owner: Misha (Database & Geospatial Pipeline Lead)
Handles GeoTIFF parsing, CRS validation, Affine transforms, band math, SAR calibration, and GeoJSON export.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import xy
    from rasterio.features import shapes
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from shapely.geometry import shape, mapping, Polygon
from core.schemas import ModalityType, InputImageMetadata


class GeospatialEngine:
    """
    Robust Geospatial & Multimodal Ingestion Engine.
    Handles GeoTIFF (Cartosat, Sentinel, RISAT) and benchmark PNG/JPEG inputs.
    """

    @staticmethod
    def inspect_and_load(file_path: Path) -> Tuple[InputImageMetadata, np.ndarray, Optional[Any]]:
        """
        Inspects input file, parses spatial metadata, and returns normalized numpy array.
        Returns: (metadata, raster_array, affine_transform)
        """
        ext = file_path.suffix.lower()

        if ext in [".tif", ".tiff", ".geotiff"] and RASTERIO_AVAILABLE:
            with rasterio.open(file_path) as src:
                crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
                bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                width = src.width
                height = src.height
                bands = src.count
                res_x, res_y = abs(src.res[0]), abs(src.res[1])
                spatial_resolution = round((res_x + res_y) / 2.0, 2)
                affine_transform = src.transform

                data = src.read()

                name_lower = file_path.name.lower()
                if "sar" in name_lower or "risat" in name_lower or "s1" in name_lower or bands == 2:
                    modality = ModalityType.SAR_C_BAND
                elif bands >= 4:
                    modality = ModalityType.OPTICAL_MULTISPECTRAL
                else:
                    modality = ModalityType.OPTICAL_RGB

                metadata = InputImageMetadata(
                    filename=file_path.name,
                    format="GeoTIFF",
                    modality=modality,
                    crs=crs_str,
                    width=width,
                    height=height,
                    bands=bands,
                    spatial_resolution_m=spatial_resolution,
                    bounding_box=bounds,
                    co_registered=True
                )
                return metadata, data, affine_transform

        else:
            img = Image.open(file_path)
            data = np.array(img)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            elif data.ndim == 3:
                data = np.transpose(data, (2, 0, 1))

            bands, height, width = data.shape
            metadata = InputImageMetadata(
                filename=file_path.name,
                format=ext.replace(".", "").upper(),
                modality=ModalityType.BENCHMARK_IMAGE,
                crs="EPSG:4326",
                width=width,
                height=height,
                bands=bands,
                spatial_resolution_m=10.0,
                bounding_box=[77.10, 28.60, 77.25, 28.75],
                co_registered=True
            )
            return metadata, data, None

    @staticmethod
    def calculate_spectral_indices(optical_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculates NDVI and NDWI if multispectral bands exist.
        """
        indices = {}
        if optical_data.shape[0] >= 4:
            green = optical_data[1].astype(np.float32)
            red = optical_data[2].astype(np.float32)
            nir = optical_data[3].astype(np.float32)
            eps = 1e-6
            indices["NDVI"] = np.clip((nir - red) / (nir + red + eps), -1.0, 1.0)
            indices["NDWI"] = np.clip((green - nir) / (green + nir + eps), -1.0, 1.0)
        return indices

    @staticmethod
    def calibrate_and_filter_sar(sar_data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Converts SAR Digital Numbers (DN) to Sigma0 (dB) backscatter and applies speckle filter.
        """
        from scipy.ndimage import uniform_filter
        sar_float = sar_data[0].astype(np.float32)
        eps = 1e-6
        sigma0_db = 10.0 * np.log10(np.square(sar_float) + eps)
        return uniform_filter(sigma0_db, size=kernel_size)

    @staticmethod
    def raster_mask_to_geojson(
        binary_mask: np.ndarray,
        affine_transform: Any,
        layer_name: str = "detected_features",
        pixel_size_m: float = 10.0
    ) -> Dict[str, Any]:
        """
        Converts a 2D binary numpy mask into a GeoJSON FeatureCollection using Affine transform.
        """
        features = []
        total_pixels = int(np.sum(binary_mask > 0))
        area_hectares = round((total_pixels * (pixel_size_m ** 2)) / 10000.0, 2)

        if RASTERIO_AVAILABLE and affine_transform is not None:
            mask_uint8 = (binary_mask > 0).astype(np.uint8)
            for geom, val in shapes(mask_uint8, mask=(mask_uint8 == 1), transform=affine_transform):
                poly = shape(geom)
                if poly.is_valid and poly.area > 0:
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "layer": layer_name,
                            "area_hectares": round((poly.area * (pixel_size_m ** 2)) / 10000.0, 2)
                        },
                        "geometry": mapping(poly)
                    })

        return {
            "layer_name": layer_name,
            "feature_type": "FeatureCollection",
            "feature_count": len(features),
            "geojson": {
                "type": "FeatureCollection",
                "features": features
            },
            "metrics": {
                "total_pixel_count": total_pixels,
                "total_area_hectares": area_hectares,
                "pixel_resolution_m": pixel_size_m
            }
        }
