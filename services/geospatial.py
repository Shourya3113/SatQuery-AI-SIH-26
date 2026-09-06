"""
SatQuery AI - Geospatial Engine & Multimodal Processing Pipeline
Owner: Misha (Database & Geospatial Pipeline Lead) & Peter (Integration Lead)

Provides:
- GeoTIFF / COG ingestion into raw and model-ready (C, H, W) slices.
- Affine coordinate projections from pixel masks to EPSG:4326 GeoJSON with metric hectare areas.
- Radiometric calibration (sigma0 dB) and adaptive Lee speckle filtering.
- Spectral indices (NDVI, NDWI).
- Day-1 mock tensors.
- GeospatialEngine wrapper for high-level pipeline orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.features import shapes as rio_shapes
    from rasterio.transform import xy
    from rasterio.warp import transform_geom

    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    CRS = None
    rio_shapes = None
    transform_geom = None
    RASTERIO_AVAILABLE = False

try:
    from shapely.geometry import box, mapping, shape as shapely_shape, Polygon

    _HAVE_SHAPELY = True
except ImportError:
    shapely_shape = None
    _HAVE_SHAPELY = False

from core.schemas import ModalityType, InputImageMetadata


# ---------------------------------------------------------------------------
# Deliverable 1 — Mock contract
# ---------------------------------------------------------------------------


def generate_mock_tensors(size: int = 512, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic model inputs so PyTorch forward passes can be built before real rasters exist."""
    rng = np.random.default_rng(seed)
    optical = rng.random((4, size, size), dtype=np.float32)
    sar = rng.uniform(-30.0, 0.0, (1, size, size)).astype(np.float32)
    return optical, sar


# ---------------------------------------------------------------------------
# Deliverable 2 — GeoTIFF / COG ingestion into (C, H, W) array slices
# ---------------------------------------------------------------------------


@dataclass
class RasterData:
    """One ingested raster: raw float32 (C, H, W) arrays + metadata."""

    path: str
    arrays: np.ndarray | None = None
    band_names: list[str] = field(default_factory=list)
    crs: str | None = None
    crs_epsg: int | None = None
    gsd_x: float | None = None
    gsd_y: float | None = None
    width: int | None = None
    height: int | None = None
    bounds: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    problems: list[str] = field(default_factory=list)
    real: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "crs": self.crs,
            "crs_epsg": self.crs_epsg,
            "gsd_x": self.gsd_x,
            "gsd_y": self.gsd_y,
            "width": self.width,
            "height": self.height,
            "bands": self.band_names,
            "band_count": len(self.band_names),
            "bounds": self.bounds,
            "metadata": self.metadata,
            "ok": self.ok,
            "problems": self.problems,
            "real": self.real,
        }


def _band_descriptions(ds: Any) -> list[str]:
    descs = ds.descriptions or []
    return [("" if d is None else str(d)).strip() for d in descs]


def load_raster(path: str) -> RasterData:
    """Open a GeoTIFF/COG and return raw float32 (C, H, W) arrays plus full metadata."""
    data = RasterData(path=path)
    p = Path(path)
    if not p.exists():
        data.problems.append(f"file not found: {path}")
        return data
    if not RASTERIO_AVAILABLE:
        data.problems.append("rasterio not available in environment")
        return data
    try:
        with rasterio.open(path) as ds:
            data.arrays = ds.read().astype(np.float32)
            data.band_names = _band_descriptions(ds)
            data.width, data.height = ds.width, ds.height
            data.real = True
            if ds.crs:
                data.crs = ds.crs.to_string()
                data.crs_epsg = ds.crs.to_epsg()
            t = ds.transform
            if t and t.a:
                data.gsd_x = abs(float(t.a))
                data.gsd_y = abs(float(t.e))
            b = ds.bounds
            data.bounds = [float(b.left), float(b.bottom), float(b.right), float(b.top)]
            data.metadata = dict(ds.tags() or {})
            data.ok = True
    except Exception as e:
        data.problems.append(f"failed to open {path}: {e}")
    return data


def normalize_bands(arrays: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0, 1] (constant bands -> zeros)."""
    arr = np.asarray(arrays, dtype=np.float32)
    mn = arr.min(axis=(-2, -1), keepdims=True)
    mx = arr.max(axis=(-2, -1), keepdims=True)
    return (arr - mn) / (mx - mn + 1e-9)


def raster_model_input(path: str, modality: str) -> Tuple[np.ndarray, RasterData]:
    """Model-ready tensor for one raster, matching the mock contract's layout."""
    data = load_raster(path)
    if data.arrays is None:
        raise ValueError(f"cannot build model input from {path}: {data.problems}")
    if modality == "sar":
        tensor = np.stack([calibrate_sigma0(band) for band in data.arrays])
    else:
        tensor = normalize_bands(data.arrays)
    return tensor.astype(np.float32), data


# ---------------------------------------------------------------------------
# Deliverable 4 — SAR radiometric calibration + adaptive Lee speckle filter
# ---------------------------------------------------------------------------


def calibrate_sigma0(dn: np.ndarray, k_cal_db: float = 0.0, eps: float = 1e-6) -> np.ndarray:
    """SAR radiometric calibration to backscatter in dB: sigma0 (dB) = 10 * log10(DN^2 + eps) - K_cal."""
    arr = np.asarray(dn, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(arr * arr + eps) - k_cal_db
    return db.astype(np.float32)


def adaptive_lee_filter(array: np.ndarray, window_size: int = 5, enl: float = 1.0) -> np.ndarray:
    """Adaptive Lee speckle filter on SAR intensity, preserving edges."""
    arr = np.asarray(array, dtype=np.float64)
    k = int(window_size)
    if k < 3 or k % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    pad = k // 2

    pad_width = [(0, 0)] * (arr.ndim - 2) + [(pad, pad), (pad, pad)]
    padded = np.pad(arr, pad_width, mode="edge")

    win = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(-2, -1))
    mean = win.mean(axis=(-2, -1))
    sq_mean = (win * win).mean(axis=(-2, -1))
    var = np.clip(sq_mean - mean * mean, 0.0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        ci2 = var / (mean * mean + 1e-12)
        cu2 = 1.0 / float(enl)
        w = np.clip(1.0 - cu2 / ci2, 0.0, 1.0)
    w = np.nan_to_num(w, nan=0.0)

    out = mean + w * (arr - mean)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Deliverable 4 — Spectral indices (NDVI / NDWI)
# ---------------------------------------------------------------------------

_RED_HINTS = ("B04", "red", "r")
_GREEN_HINTS = ("B03", "green", "g")
_NIR_HINTS = ("B08", "B8A", "B8", "nir", "n")


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Generic band-ratio index (a - b) / (a + b), float32, safe at zeros."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b + eps)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    return normalized_difference(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index: (Green - NIR) / (Green + NIR)."""
    return normalized_difference(green, nir)


def _first_band_by_hint(band_names: Iterable[str], hints: tuple[str, ...]) -> int | None:
    for i, name in enumerate(band_names):
        low = name.lower()
        if any(hint.lower() in low for hint in hints):
            return i
    return None


def raster_index(path: str, index: str = "ndvi") -> Tuple[np.ndarray, dict]:
    """Compute NDVI/NDWI straight from a multispectral GeoTIFF."""
    data = load_raster(path)
    if data.arrays is None:
        raise ValueError(f"cannot compute {index} from {path}: {data.problems}")
    names = data.band_names or [f"band{i + 1}" for i in range(data.arrays.shape[0])]

    index = index.lower().replace("-", "")
    if index == "ndvi":
        nir_i = _first_band_by_hint(names, _NIR_HINTS)
        red_i = _first_band_by_hint(names, _RED_HINTS)
        if nir_i is None or red_i is None:
            raise ValueError(f"cannot resolve NIR/red bands for NDVI from {names}")
        arr = ndvi(data.arrays[nir_i], data.arrays[red_i])
        info = {"formula": "NDVI = (NIR - Red) / (NIR + Red)", "nir_band": names[nir_i], "red_band": names[red_i]}
    elif index == "ndwi":
        g_i = _first_band_by_hint(names, _GREEN_HINTS)
        nir_i = _first_band_by_hint(names, _NIR_HINTS)
        if g_i is None or nir_i is None:
            raise ValueError(f"cannot resolve green/NIR bands for NDWI from {names}")
        arr = ndwi(data.arrays[g_i], data.arrays[nir_i])
        info = {"formula": "NDWI = (Green - NIR) / (Green + NIR)", "green_band": names[g_i], "nir_band": names[nir_i]}
    else:
        raise ValueError(f"unknown index {index!r}; expected 'ndvi' or 'ndwi'")
    return arr, info


# ---------------------------------------------------------------------------
# Deliverable 3 — Affine coordinate projector: mask (H, W) -> GeoJSON
# ---------------------------------------------------------------------------


def _auto_utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6) + 1
    zone = max(1, min(60, zone))
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _ring_area(ring: list) -> float:
    n = len(ring)
    if n < 4:
        return 0.0
    return 0.5 * sum(
        ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1] for i in range(n)
    )


def _geom_area(geom: dict) -> float:
    gtype = geom.get("type")
    if gtype == "Polygon":
        return abs(sum(_ring_area(r) for r in geom.get("coordinates", [])))
    if gtype == "MultiPolygon":
        return sum(abs(sum(_ring_area(r) for r in poly)) for poly in geom.get("coordinates", []))
    return 0.0


def metric_area_m2(geom: dict, crs: str | None = None) -> Tuple[float, str]:
    """Planar area of geometry in square metres, measured before degree reprojection."""
    if crs and RASTERIO_AVAILABLE:
        from rasterio.crs import CRS

        c = CRS.from_user_input(crs)
        if c.is_geographic:
            (x0, y0) = list(geom.get("coordinates")[0])[0]
            utm = _auto_utm_epsg(x0, y0)
            geom_m = transform_geom(crs, utm, geom)
            return float(_geom_area(geom_m)), f"measured in {utm} after {crs}->{utm}"
    return float(_geom_area(geom)), "measured in source CRS map units (assumed metres)"


def mask_to_geojson(
    mask: np.ndarray,
    transform: Any,
    crs: str | None = None,
    mask_value: int = 1,
    label: str = "region",
    simplify_tol: float = 1e-4,
    min_area_ha: float = 0.0,
    target_crs: str = "EPSG:4326",
) -> dict:
    """Project a binary prediction mask (H, W) into a GeoJSON FeatureCollection."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W); got shape {arr.shape}")
    if transform is None:
        raise ValueError("an affine transform (e.g. src.transform) is required")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    features: list[dict] = []
    note: list[str] = []
    total_m2 = 0.0
    n_raw = 0

    if RASTERIO_AVAILABLE:
        for geom, val in rio_shapes(arr, mask=(arr == mask_value), transform=transform):
            if float(val) != float(mask_value):
                continue
            n_raw += 1
            area_m2, area_note = metric_area_m2(geom, crs)
            if area_m2 < min_area_ha * 1e4:
                continue

            if crs and crs != target_crs and transform_geom:
                geom_out = transform_geom(crs, target_crs, geom)
            else:
                geom_out = geom

            if _HAVE_SHAPELY and simplify_tol is not None and simplify_tol > 0:
                geom_out = shapely_shape(geom_out).simplify(simplify_tol, preserve_topology=True).__geo_interface__
            elif n_raw == 1:
                note.append(f"shapely not installed or simplify skipped (tol={simplify_tol})")

            total_m2 += area_m2
            props: dict[str, Any] = {
                "value": int(val),
                "label": label,
                "area_m2": round(area_m2, 4),
                "area_ha": round(area_m2 / 1e4, 6),
                "area_km2": round(area_m2 / 1e6, 8),
                "measurement": area_note,
            }
            features.append({"type": "Feature", "properties": props, "geometry": geom_out})

    if not features and n_raw:
        note.append(f"all {n_raw} candidate polygon(s) below min_area_ha={min_area_ha}")

    return {
        "type": "FeatureCollection",
        "name": label,
        "features": features,
        "satquery": {
            "source_crs": crs,
            "target_crs": target_crs,
            "simplify_tol": simplify_tol,
            "n_features": len(features),
            "n_candidate_polygons": n_raw,
            "total_area_ha": round(total_m2 / 1e4, 6),
            "total_area_km2": round(total_m2 / 1e6, 8),
            "note": "; ".join(note) if note else "half-pixel aligned via rasterio affine transform",
        },
    }


def raster_mask_to_geojson(mask: np.ndarray, source: Any, **kwargs: Any) -> dict:
    """Convenience function that accepts either a GeoTIFF path or an affine transform."""
    if isinstance(source, (str, Path)):
        if RASTERIO_AVAILABLE:
            with rasterio.open(str(source)) as ds:
                crs = ds.crs.to_string() if ds.crs else None
                transform = ds.transform
            return mask_to_geojson(mask, transform=transform, crs=crs, **kwargs)
        else:
            return mask_to_geojson(mask, transform=None, **kwargs)
    else:
        return mask_to_geojson(mask, transform=source, **kwargs)


def pixel_to_world(transform: Any, row: float, col: float) -> Tuple[float, float]:
    """Affine 6-parameter projection of pixel center (row, col) to world (x, y)."""
    c, a, b, f, d, e = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
    x = a * (col + 0.5) + b * (row + 0.5) + c
    y = d * (col + 0.5) + e * (row + 0.5) + f
    return float(x), float(y)


def world_to_pixel(transform: Any, x: float, y: float) -> Tuple[float, float]:
    """Inverse affine: world (x, y) -> fractional (row, col) (0-based)."""
    c, a, b, f, d, e = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
    det = a * e - b * d
    if det == 0:
        raise ValueError("singular affine transform")
    col = (e * (x - c) - b * (y - f)) / det - 0.5
    row = (a * (y - f) - d * (x - c)) / det - 0.5
    return float(row), float(col)


# ---------------------------------------------------------------------------
# High-Level GeospatialEngine Class (Orchestrator Contract)
# ---------------------------------------------------------------------------


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
        p = Path(file_path)
        ext = p.suffix.lower()

        if ext in [".tif", ".tiff", ".geotiff"] and RASTERIO_AVAILABLE:
            with rasterio.open(p) as src:
                crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
                bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                width = src.width
                height = src.height
                bands = src.count
                res_x, res_y = abs(src.res[0]), abs(src.res[1])
                spatial_resolution = round((res_x + res_y) / 2.0, 2)
                affine_transform = src.transform

                data = src.read()

                name_lower = p.name.lower()
                if "sar" in name_lower or "risat" in name_lower or "s1" in name_lower or bands == 2:
                    modality = ModalityType.SAR_C_BAND
                elif bands >= 4:
                    modality = ModalityType.OPTICAL_MULTISPECTRAL
                else:
                    modality = ModalityType.OPTICAL_RGB

                metadata = InputImageMetadata(
                    filename=p.name,
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
            img = Image.open(p)
            data = np.array(img)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            elif data.ndim == 3:
                data = np.transpose(data, (2, 0, 1))

            bands, height, width = data.shape
            metadata = InputImageMetadata(
                filename=p.name,
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
        """Calculates NDVI and NDWI if multispectral bands exist."""
        indices = {}
        if optical_data.shape[0] >= 4:
            green = optical_data[1].astype(np.float32)
            red = optical_data[2].astype(np.float32)
            nir = optical_data[3].astype(np.float32)
            indices["NDVI"] = ndvi(nir, red)
            indices["NDWI"] = ndwi(green, nir)
        return indices

    @staticmethod
    def calibrate_and_filter_sar(sar_data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Converts SAR DN to Sigma0 (dB) backscatter and applies adaptive Lee filter."""
        sar_float = sar_data[0].astype(np.float32)
        sigma0_db = calibrate_sigma0(sar_float)
        return adaptive_lee_filter(sigma0_db, window_size=kernel_size)

    @staticmethod
    def raster_mask_to_geojson(
        binary_mask: np.ndarray,
        affine_transform: Any,
        layer_name: str = "detected_features",
        pixel_size_m: float = 10.0
    ) -> Dict[str, Any]:
        """Converts a 2D binary numpy mask into a GeoJSON FeatureCollection."""
        features = []
        total_pixels = int(np.sum(binary_mask > 0))
        area_hectares = round((total_pixels * (pixel_size_m ** 2)) / 10000.0, 2)

        if RASTERIO_AVAILABLE and affine_transform is not None:
            mask_uint8 = (binary_mask > 0).astype(np.uint8)
            for geom, val in rio_shapes(mask_uint8, mask=(mask_uint8 == 1), transform=affine_transform):
                poly = shapely_shape(geom) if _HAVE_SHAPELY else None
                if poly and poly.is_valid and poly.area > 0:
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
