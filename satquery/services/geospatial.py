"""Geospatial + SAR pipeline — Track 5 (database & geospatial lead).

Everything downstream tracks consume from this module:

- ``generate_mock_tensors`` — the Day-1 mock contract so model work starts
  before real rasters exist (4-band optical 0..1 + calibrated SAR dB).
- ``load_raster`` / ``raster_model_input`` — GeoTIFF/COG ingestion into raw or
  model-ready ``(C, H, W)`` float32 slices plus the ``InputImageMetadata``-style
  contract (CRS, GSD, bounds, affine transform, band names).
- ``mask_to_geojson`` — deterministic affine projection of a binary ``(H, W)``
  prediction mask into an EPSG:4326 GeoJSON ``FeatureCollection`` with metric
  hectare areas. No LLM coordinate hallucination: every vertex comes from the
  raster's 6-parameter affine transform via ``rasterio.features.shapes`` (which
  applies the PixelIsArea half-pixel offset natively).
- Spectral index + SAR radiometry math: NDVI, NDWI, sigma0 dB calibration and
  the adaptive Lee speckle filter.

Requires numpy + rasterio. shapely is optional and only used for
topology-preserving Douglas-Peucker simplification of the final GeoJSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform_geom

try:
    from shapely.geometry import shape as shapely_shape

    _HAVE_SHAPELY = True
except ImportError:  # pragma: no cover - exercised on machines without shapely
    shapely_shape = None
    _HAVE_SHAPELY = False

# ---------------------------------------------------------------------------
# Deliverable 1 — mock contract (unblocks model forward passes immediately)
# ---------------------------------------------------------------------------


def generate_mock_tensors(size: int = 512, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic model inputs so Chhavi's PyTorch forward passes can be built
    before real GeoTIFF parsing exists.

    Returns ``(optical, sar)``:
      * optical: ``(4, size, size)`` float32, bands B02/B03/B04/B08 (RGB+NIR),
        each pixel uniform in [0, 1] — what the fine-tuned VLM consumes.
      * sar:     ``(1, size, size)`` float32, calibrated backscatter sigma0 in
        dB within [-30, 0] (single VV-like channel).

    ``seed`` makes the pair reproducible.
    """
    rng = np.random.default_rng(seed)
    optical = rng.random((4, size, size), dtype=np.float32)
    sar = rng.uniform(-30.0, 0.0, (1, size, size)).astype(np.float32)
    return optical, sar


# ---------------------------------------------------------------------------
# Deliverable 2 — GeoTIFF / COG ingestion into (C, H, W) array slices
# ---------------------------------------------------------------------------


@dataclass
class RasterData:
    """One ingested raster: raw float32 ``(C, H, W)`` arrays + all metadata the
    trace and handshake contracts need (InputImageMetadata equivalent)."""

    path: str
    arrays: np.ndarray | None = None        # (C, H, W) float32, raw digital numbers
    band_names: list[str] = field(default_factory=list)
    crs: str | None = None
    crs_epsg: int | None = None
    gsd_x: float | None = None
    gsd_y: float | None = None
    width: int | None = None
    height: int | None = None
    bounds: list[float] | None = None       # [left, bottom, right, top]
    metadata: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    problems: list[str] = field(default_factory=list)
    real: bool = False

    def to_dict(self) -> dict:
        """JSON-safe metadata (arrays deliberately excluded — registry stores
        this, model code reads arrays directly from the returned object)."""
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
    """Open a GeoTIFF/COG and return raw float32 ``(C, H, W)`` arrays plus full
    metadata (CRS/EPSG, GSD, bounds, affine-derived fields, band names)."""
    from pathlib import Path

    data = RasterData(path=path)
    if not Path(path).exists():
        data.problems.append(f"file not found: {path}")
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
    except Exception as e:  # noqa: BLE001 - report any open failure as a problem
        data.problems.append(f"failed to open {path}: {e}")
    return data


def normalize_bands(arrays: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0, 1] (constant bands -> zeros)."""
    arr = np.asarray(arrays, dtype=np.float32)
    mn = arr.min(axis=(-2, -1), keepdims=True)
    mx = arr.max(axis=(-2, -1), keepdims=True)
    return (arr - mn) / (mx - mn + 1e-9)


def raster_model_input(path: str, modality: str) -> tuple[np.ndarray, RasterData]:
    """Model-ready tensor for one raster, matching the mock contract's layout.

    * ``modality="optical"``: min-max normalized ``(C, H, W)`` float32 in [0, 1].
    * ``modality="sar"``:     radiometrically calibrated ``sigma0`` dB per band.

    Returns ``(tensor, meta)`` — meta is the same RasterData for the trace.
    """
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
    """SAR radiometric calibration to backscatter in dB:

        sigma0 (dB) = 10 * log10(DN^2 + eps) - K_cal

    ``DN`` is the (amplitude) digital number; ``k_cal_db`` is the product
    calibration constant (often stored in the GRD product metadata). Output is
    float32; calm water lands far below -18 dB, built-up double-bounce above
    -6 dB for real calibrated products.
    """
    arr = np.asarray(dn, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(arr * arr + eps) - k_cal_db
    return db.astype(np.float32)


def adaptive_lee_filter(array: np.ndarray, window_size: int = 5, enl: float = 1.0) -> np.ndarray:
    """Adaptive Lee speckle filter (default 5x5 window) on SAR intensity.

    Replaces destructive Gaussian blur: the local mean/variance decide how much
    to smooth, so homogeneous open water is flattened while sharp structural and
    coastline edges are preserved.

        R_hat = mean + W * (center - mean),   W = max(0, 1 - Cu^2 / Ci^2)

    with ``Cu^2 = 1 / ENL`` (speckle coefficient of variation) and ``Ci^2`` the
    local coefficient of variation (variance / mean^2). Accepts 2D ``(H, W)``
    or 3D ``(C, H, W)`` arrays and filters every band.
    """
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

    # Coefficient of variation of the local window; guard division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        ci2 = var / (mean * mean + 1e-12)
        cu2 = 1.0 / float(enl)
        w = np.clip(1.0 - cu2 / ci2, 0.0, 1.0)
    w = np.nan_to_num(w, nan=0.0)

    out = mean + w * (arr - mean)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Deliverable 4 — spectral indices (NDVI / NDWI)
# ---------------------------------------------------------------------------

# Band-name hints used to resolve spectral roles from GeoTIFF descriptions.
_RED_HINTS = ("B04", "red", "r")
_GREEN_HINTS = ("B03", "green", "g")
_NIR_HINTS = ("B08", "B8A", "B8", "nir", "n")


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Generic band-ratio index ``(a - b) / (a + b)``, float32, safe at zeros."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b + eps)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index (NIR - Red) / (NIR + Red)."""
    return normalized_difference(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (Green - NIR) / (Green + NIR)."""
    return normalized_difference(green, nir)


def _first_band_by_hint(band_names: Iterable[str], hints: tuple[str, ...]) -> int | None:
    """0-based index of the first band whose description matches a hint."""
    for i, name in enumerate(band_names):
        low = name.lower()
        if any(hint.lower() in low for hint in hints):
            return i
    return None


def raster_index(path: str, index: str = "ndvi") -> tuple[np.ndarray, dict]:
    """Compute NDVI/NDWI straight from a multispectral GeoTIFF by resolving
    bands from the file's descriptions (B04/B03/B08 or red/green/nir aliases).

    Returns ``(array (H, W) float32, info)`` where info records which bands were
    used — that mapping goes into the auditable trace.
    """
    data = load_raster(path)
    if data.arrays is None:
        raise ValueError(f"cannot compute {index} from {path}: {data.problems}")
    names = data.band_names or [f"band{i + 1}" for i in range(data.arrays.shape[0])]

    index = index.lower().replace("-", "")
    if index == "ndvi":
        nir_i, red_i = _first_band_by_hint(names, _NIR_HINTS), _first_band_by_hint(names, _RED_HINTS)
        if nir_i is None or red_i is None:
            raise ValueError(f"cannot resolve NIR/red bands for NDVI from {names}")
        arr = ndvi(data.arrays[nir_i], data.arrays[red_i])
        info = {"formula": "NDVI = (NIR - Red) / (NIR + Red)", "nir_band": names[nir_i], "red_band": names[red_i]}
    elif index == "ndwi":
        g_i, nir_i = _first_band_by_hint(names, _GREEN_HINTS), _first_band_by_hint(names, _NIR_HINTS)
        if g_i is None or nir_i is None:
            raise ValueError(f"cannot resolve green/NIR bands for NDWI from {names}")
        arr = ndwi(data.arrays[g_i], data.arrays[nir_i])
        info = {"formula": "NDWI = (Green - NIR) / (Green + NIR)", "green_band": names[g_i], "nir_band": names[nir_i]}
    else:
        raise ValueError(f"unknown index {index!r}; expected 'ndvi' or 'ndwi'")
    return arr, info


# ---------------------------------------------------------------------------
# Deliverable 3 — affine coordinate projector: mask (H, W) -> GeoJSON
# ---------------------------------------------------------------------------


def _auto_utm_epsg(lon: float, lat: float) -> str:
    """Best-effort UTM zone for metric area measurement of geographic rasters."""
    zone = int((lon + 180.0) // 6) + 1
    zone = max(1, min(60, zone))
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _ring_area(ring: list) -> float:
    """Signed shoelace area of one (closed) coordinate ring in its own units."""
    n = len(ring)
    if n < 4:
        return 0.0
    return 0.5 * sum(
        ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1] for i in range(n)
    )


def _geom_area(geom: dict) -> float:
    """Absolute planar area of a GeoJSON polygon (handles holes + multipolygons)."""
    gtype = geom.get("type")
    if gtype == "Polygon":
        return abs(sum(_ring_area(r) for r in geom.get("coordinates", [])))
    if gtype == "MultiPolygon":
        return sum(abs(sum(_ring_area(r) for r in poly)) for poly in geom.get("coordinates", []))
    return 0.0


def metric_area_m2(geom: dict, crs: str | None = None) -> tuple[float, str]:
    """Planar area of a geometry in square metres, measured BEFORE reprojection
    to degrees so high-latitude distortion never leaks into hectare counts.

    * Projected CRS (UTM etc.): shoelace directly in map units (assumed metres).
    * Geographic CRS: reproject the geometry to an auto-derived UTM zone first.

    Returns ``(area_m2, note)``.
    """
    if crs:
        from rasterio.crs import CRS

        c = CRS.from_user_input(crs)
        if c.is_geographic:
            (x0, y0) = list(geom.get("coordinates")[0])[0]  # first vertex as lon/lat proxy
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
    """Deterministically project a binary prediction mask ``(H, W)`` into a
    GeoJSON ``FeatureCollection`` in ``target_crs`` (EPSG:4326 for MapLibre).

    Half-pixel alignment: ``rasterio.features.shapes`` applies the raster's
    affine transform to pixel *centers* (x + 0.5, y + 0.5 vs. the corner
    origin), so the returned polygons share edges with the source raster
    exactly — the caller just passes ``transform=src.transform``.

    Per-feature properties carry ``area_ha`` / ``area_km2`` (metric areas
    computed in the projected source CRS before any degree reprojection) and a
    foreign member ``"satquery"`` holds the totals + provenance for the trace.
    """
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

    for geom, val in rio_shapes(arr, mask=(arr == mask_value), transform=transform):
        if float(val) != float(mask_value):
            continue
        n_raw += 1
        area_m2, area_note = metric_area_m2(geom, crs)
        if area_m2 < min_area_ha * 1e4:
            continue

        if crs and crs != target_crs:
            geom_out = transform_geom(crs, target_crs, geom)
        else:
            geom_out = geom

        if _HAVE_SHAPELY and simplify_tol is not None and simplify_tol > 0:
            geom_out = shapely_shape(geom_out).simplify(simplify_tol, preserve_topology=True).__geo_interface__
        elif n_raw == 1:
            note.append(f"shapely not installed; topology-preserving simplify skipped (tol={simplify_tol})")

        # Centroid from the (possibly simplified) output ring — first vertex ok
        # for tests, but prefer polygon interior point when shapely exists.
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


def raster_mask_to_geojson(mask: np.ndarray, geotiff_path: str, **kwargs: Any) -> dict:
    """``mask_to_geojson`` convenience that pulls transform + CRS from a raster
    file (the Achintya -> Misha handshake: predicted masks -> projected layers)."""
    with rasterio.open(geotiff_path) as ds:
        crs = ds.crs.to_string() if ds.crs else None
        transform = ds.transform
    return mask_to_geojson(mask, transform=transform, crs=crs, **kwargs)


def pixel_to_world(transform: Any, row: float, col: float) -> tuple[float, float]:
    """Affine 6-parameter projection of a pixel *center* (row, col) to world
    ``(x, y)``. Equivalent to ``rasterio.xy(transform, row, col)``.
    """
    c, a, b, f, d, e = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
    x = a * (col + 0.5) + b * (row + 0.5) + c
    y = d * (col + 0.5) + e * (row + 0.5) + f
    return float(x), float(y)


def world_to_pixel(transform: Any, x: float, y: float) -> tuple[float, float]:
    """Inverse affine: world ``(x, y)`` -> fractional ``(row, col)`` (0-based)."""
    c, a, b, f, d, e = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
    det = a * e - b * d
    if det == 0:
        raise ValueError("singular affine transform")
    col = (e * (x - c) - b * (y - f)) / det - 0.5
    row = (a * (y - f) - d * (x - c)) / det - 0.5
    return float(row), float(col)
