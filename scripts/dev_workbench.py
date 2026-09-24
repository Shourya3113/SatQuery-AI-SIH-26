"""
SatQuery AI - Geospatial Dev Workbench (single file, no frontend required)
Owner: Misha (Database & Geospatial Pipeline Lead)

Runs the REAL geospatial pipeline on a raster and renders the result as one
self-contained interactive map, so you can *see* whether the affine projection is
correct instead of reading coordinate arrays.

Because every layer is placed from the raster's own 6-parameter affine transform,
a broken transform shows up instantly as visibly separated layers:

    - raster footprint (from the transform's bounds, reprojected to EPSG:4326)
    - georeferenced preview of the source pixels (stretched, same bounds)
    - the polygons produced by mask_to_geojson

Usage
-----
    python scripts/dev_workbench.py                              # optical sample, NDVI > 0.2
    python scripts/dev_workbench.py --index ndwi --threshold 0.1
    python scripts/dev_workbench.py --mode sar                   # sigma0 -> Lee -> calm water
    python scripts/dev_workbench.py --mode sar --regime volume
    python scripts/dev_workbench.py --raster path/to.tif --mode sar --k-cal 83
    python scripts/dev_workbench.py --open                       # open in the browser

Outputs (gitignored, under data/outputs/):
    dev_workbench/<stem>_<mode>.html   self-contained map (open in any browser)
    dev_workbench/<stem>_<mode>.json   raw payload, for inspecting data structures

Note: masks here are thresholded straight from the raster, NOT model predictions.
This harness verifies projection/vectorisation, not model quality.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image

# Add project root to sys.path (same bootstrap as run_benchmarks.py)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services import geospatial as gtx

try:
    import folium

    _HAVE_FOLIUM = True
except ImportError:  # pragma: no cover - exercised only without folium installed
    folium = None
    _HAVE_FOLIUM = False

SAMPLE_DIR = ROOT_DIR / "data" / "samples"
OUT_DIR = ROOT_DIR / "data" / "outputs" / "dev_workbench"

# Scattering regimes exposed by --regime, mapped onto geospatial.scattering_regime labels.
_REGIMES = {
    "surface": (0, "surface / specular  (sigma0 < -16 dB, calm water)"),
    "volume": (1, "volume / diffuse    (-16 .. -6 dB, canopy, crops)"),
    "double": (2, "double-bounce       (sigma0 > -6 dB, urban)"),
}


# ---------------------------------------------------------------------------
# Mask builders (pure -- importable and testable without folium)
# ---------------------------------------------------------------------------


def looks_like_db(band: np.ndarray) -> bool:
    """True when a SAR band is already calibrated backscatter in dB.

    Amplitude/DN rasters are non-negative, so any negative floor means the data is
    already in dB and must NOT be calibrated a second time. Same heuristic as
    tools/fusion_engine.py so the harness and the pipeline agree.
    """
    return float(np.min(band)) < -5.0 and float(np.max(band)) < 25.0


def build_optical_mask(
    path: str, index: str = "ndvi", threshold: float = 0.2
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Threshold a spectral index into a binary mask using the real band resolver."""
    arr, info = gtx.raster_index(path, index)
    mask = (arr > float(threshold)).astype(np.uint8)
    params = {
        "mode": "optical",
        "index": index.upper(),
        "threshold": float(threshold),
        "comparison": ">",
        "formula": info["formula"],
        "band_resolution": {k: v for k, v in info.items() if k.endswith("_band") or k == "band_indices"},
        "index_range": [round(float(arr.min()), 4), round(float(arr.max()), 4)],
    }
    return mask, params


def build_sar_mask(
    path: str, regime: str = "surface", k_cal_db: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Calibrate (only if needed) -> Lee filter -> select one scattering regime."""
    data = gtx.load_raster(path)
    if data.arrays is None:
        raise SystemExit(f"could not load {path}: {data.problems}")
    band = data.arrays[0]

    already_db = looks_like_db(band)
    if already_db:
        sigma0_db = band.astype(np.float32)
        calibration = "input already sigma0 dB - calibration skipped"
    else:
        k_cal = k_cal_db
        if k_cal is None:
            # Mirror fusion_engine: high-amplitude DNs imply a real GRD amplitude image.
            k_cal = gtx.SENTINEL1_GRD_K_CAL_DB if float(np.max(band)) > 100.0 else 0.0
        sigma0_db = gtx.calibrate_sigma0(band, k_cal_db=k_cal)
        calibration = f"sigma0(dB) = 10*log10(DN^2) - {k_cal} dB"

    sigma0_db = gtx.adaptive_lee_filter(sigma0_db, window_size=5)
    labels = gtx.scattering_regime(sigma0_db)
    label, description = _REGIMES[regime]
    mask = (labels == label).astype(np.uint8)

    params = {
        "mode": "sar",
        "regime": regime,
        "regime_description": description,
        "calibration": calibration,
        "already_db": bool(already_db),
        "speckle_filter": "adaptive 5x5 Lee",
        "thresholds_db": {
            "surface": gtx.SIGMA0_WATER_MAX_DB,
            "volume": gtx.SIGMA0_VOLUME_MAX_DB,
        },
        "regimes": gtx.classify_sigma0_regimes(sigma0_db),
        "sigma0_range_db": [round(float(sigma0_db.min()), 2), round(float(sigma0_db.max()), 2)],
    }
    return mask, params


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _stretch_band(band: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(band)), float(np.max(band))
    if hi - lo < 1e-9:
        return np.zeros(band.shape, dtype=np.uint8)
    return (((band - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def _rgb_preview(data: "gtx.RasterData") -> np.ndarray:
    """Best-effort RGB preview: prefer red/green/blue band hints, else first three."""
    names = data.band_names or []
    arrays = data.arrays

    def pick(hints) -> Optional[int]:
        for hint in hints:
            for i, name in enumerate(names):
                if hint.lower() in str(name).lower():
                    return i
        return None

    r, gr, b = pick(("B04", "red")), pick(("B03", "green")), pick(("B02", "blue"))
    if None in (r, gr, b) or len(set((r, gr, b))) < 3:
        r, gr, b = 0, min(1, arrays.shape[0] - 1), min(2, arrays.shape[0] - 1)
    return np.stack([_stretch_band(arrays[r]), _stretch_band(arrays[gr]), _stretch_band(arrays[b])], axis=-1)


def _single_band_preview(data: "gtx.RasterData") -> np.ndarray:
    gray = _stretch_band(data.arrays[0])
    return np.stack([gray, gray, gray], axis=-1)


def _png_data_uri(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _raster_facts(path: str) -> Dict[str, Any]:
    """Affine + footprint straight from the file: the single source of truth."""
    with rasterio.open(path) as ds:
        west, south, east, north = transform_bounds(
            ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21
        )
    return {
        "path": str(path),
        "crs": str(ds.crs),
        "bounds_source_crs": [float(v) for v in ds.bounds],
        "bounds_wgs84": [float(west), float(south), float(east), float(north)],
        "resolution": [float(ds.res[0]), float(ds.res[1])],
        "shape": [int(ds.height), int(ds.width)],
        "bands": int(ds.count),
        "transform": list(ds.transform)[:6],
    }


def _info_panel_html(payload: Dict[str, Any]) -> str:
    facts, params, gj = payload["raster"], payload["params"], payload["geojson"]
    b = facts["bounds_wgs84"]
    rows = [
        ("Source raster", Path(facts["path"]).name),
        ("CRS / resolution", f"{facts['crs']} @ {facts['resolution'][0]:g} m"),
        ("Size / bands", f"{facts['shape'][1]} x {facts['shape'][0]} px, {facts['bands']} band(s)"),
        ("Footprint (WGS84)", f"{b[0]:.5f}, {b[1]:.5f}  ->  {b[2]:.5f}, {b[3]:.5f}"),
        ("Mode", params["mode"].upper()),
    ]
    if params["mode"] == "optical":
        rows += [
            ("Index / rule", f"{params['index']} {params['comparison']} {params['threshold']}"),
            ("Bands used", json.dumps(params["band_resolution"].get("band_indices", {}))),
        ]
    else:
        rows += [
            ("Regime", params["regime_description"]),
            ("Calibration", params["calibration"]),
            ("sigma0 range", f"{params['sigma0_range_db'][0]} .. {params['sigma0_range_db'][1]} dB"),
        ]
    rows += [
        ("Mask pixels", f"{payload['mask_pixels']} / {facts['shape'][0] * facts['shape'][1]}"),
        ("Features", str(gj["satquery"]["n_features"])),
        ("Total area", f"{gj['satquery']['total_area_ha']} ha"),
        ("Output CRS", gj["satquery"]["target_crs"]),
    ]
    body = "".join(
        f"<tr><td style='padding:2px 8px 2px 0;color:#64748b'>{k}</td>"
        f"<td style='padding:2px 0;font-weight:600'>{v}</td></tr>"
        for k, v in rows
    )
    return f"""
    <div style="position:fixed;top:12px;right:12px;z-index:9999;background:#fff;
                border:1px solid #cbd5e1;border-radius:8px;padding:12px 14px;
                font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
                box-shadow:0 4px 14px rgba(15,23,42,.18);max-width:420px">
      <div style="font:600 13px system-ui,sans-serif;margin-bottom:8px">
        SatQuery dev workbench &mdash; geospatial
      </div>
      <table style="border-collapse:collapse">{body}</table>
      <div style="margin-top:8px;color:#94a3b8">
        Layers are placed from the raster affine &mdash; if they drift apart, the transform is wrong.
      </div>
    </div>
    """


def render(payload: Dict[str, Any], preview_rgb: np.ndarray, out_html: Path) -> None:
    """Write the self-contained Folium map + sibling JSON payload."""
    if not _HAVE_FOLIUM:
        raise SystemExit(
            "folium is required for the map output.\n"
            "Install it with:  uv pip install folium   (or: pip install folium)"
        )

    facts = payload["raster"]
    w, s, e, n = facts["bounds_wgs84"]
    center = [(s + n) / 2.0, (w + e) / 2.0]

    # OSM's volunteer tile servers 403 pages opened from file:// / embedded use,
    # so default to keyless providers that permit it. Esri World Imagery doubles
    # as real satellite context for judging the mask.
    fmap = folium.Map(location=center, zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        attr=("Tiles &copy; Esri &mdash; Source: Esri, Maxar, "
              "Earthstar Geographics"),
        name="Esri World Imagery",
    ).add_to(fmap)
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"),
        attr=("Tiles &copy; Esri &mdash; Source: Esri, HERE, Garmin, "
              "FAO, NOAA, USGS"),
        name="Esri World Topo (light)",
    ).add_to(fmap)

    # 1. Raster footprint, derived from the affine transform alone.
    folium.Rectangle(
        bounds=[[s, w], [n, e]],
        color="#0f172a", weight=1, dash_array="4", fill=False,
        tooltip=f"raster footprint ({facts['crs']})",
    ).add_to(fmap)

    # 2. Georeferenced source pixels, pinned to the same bounds.
    folium.raster_layers.ImageOverlay(
        image=_png_data_uri(preview_rgb),
        bounds=[[s, w], [n, e]],
        opacity=0.85,
        name="source pixels",
        interactive=True,
        cross_origin=False,
    ).add_to(fmap)

    # 3. The polygons produced by the real projector.
    gj = payload["geojson"]
    if gj["features"]:
        folium.GeoJson(
            gj,
            name=f"mask polygons ({len(gj['features'])})",
            style_function=lambda _f: {
                "color": "#dc2626", "weight": 1.5, "fillColor": "#dc2626", "fillOpacity": 0.35,
            },
            tooltip=folium.GeoJsonTooltip(fields=["label", "area_ha"], aliases=["label", "area (ha)"]),
        ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[s, w], [n, e]])

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    html = out_html.read_text(encoding="utf-8")
    html = html.replace("</body>", _info_panel_html(payload) + "\n</body>")
    out_html.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(raster: str, mode: str, index: str, threshold: float,
        regime: str, k_cal: Optional[float], min_area_ha: float,
        open_browser: bool) -> Dict[str, Any]:
    facts = _raster_facts(raster)
    data = gtx.load_raster(raster)
    if data.arrays is None:
        raise SystemExit(f"could not load {raster}: {data.problems}")
    facts.update({
        "band_names": list(data.band_names),
        "read_mode": data.metadata.get("satquery_read_mode", "full read"),
    })

    if mode == "optical":
        mask, params = build_optical_mask(raster, index=index, threshold=threshold)
        preview = _rgb_preview(data)
    else:
        mask, params = build_sar_mask(raster, regime=regime, k_cal_db=k_cal)
        preview = _single_band_preview(data)

    # The real projector, with the transform read back off the file.
    with rasterio.open(raster) as ds:
        gj = gtx.mask_to_geojson(
            mask, transform=ds.transform, crs=ds.crs.to_string(),
            label=params.get("index", params.get("regime", "region")).lower(),
            min_area_ha=float(min_area_ha),
        )
    params["min_area_ha"] = float(min_area_ha)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raster": facts,
        "params": params,
        "mask_pixels": int(mask.sum()),
        "geojson": gj,
    }
    stem = Path(raster).stem
    out_html = OUT_DIR / f"{stem}_{mode}.html"
    out_json = OUT_DIR / f"{stem}_{mode}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    render(payload, preview, out_html)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"raster      : {raster}  ({facts['crs']} @ {facts['resolution'][0]:g} m, read: {facts['read_mode']})")
    print(f"mode        : {params['mode']}")
    if params["mode"] == "optical":
        print(f"rule        : {params['index']} {params['comparison']} {params['threshold']}"
              f"  (range {params['index_range'][0]} .. {params['index_range'][1]})")
    else:
        print(f"calibration : {params['calibration']}")
        print(f"regime      : {params['regime_description']}")
        print(f"regime mix  : {params['regimes']['counts']}")
    print(f"mask pixels : {payload['mask_pixels']}")
    print(f"features    : {gj['satquery']['n_features']}  |  total {gj['satquery']['total_area_ha']} ha"
          f"  |  output {gj['satquery']['target_crs']}")
    print(f"map         : {out_html}")
    print(f"payload     : {out_json}")
    if not gj["features"]:
        print("note        : no polygons - loosen --threshold or pick another --regime")
    if open_browser:
        webbrowser.open(out_html.resolve().as_uri())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery geospatial dev workbench")
    parser.add_argument("--raster", default=str(SAMPLE_DIR / "optical.tif"),
                        help="input GeoTIFF/COG (default: the optical sample)")
    parser.add_argument("--mode", choices=("optical", "sar"), default="optical")
    parser.add_argument("--index", default="ndvi", help="optical index: ndvi/ndwi/mndwi/ndbi")
    parser.add_argument("--threshold", type=float, default=0.2, help="index threshold (mask = index > threshold)")
    parser.add_argument("--regime", choices=tuple(_REGIMES), default="surface",
                        help="which SAR scattering regime to vectorise")
    parser.add_argument("--k-cal", type=float, default=None,
                        help="override K_cal dB (default: auto - 83 for GRD-like DNs, else 0)")
    parser.add_argument("--min-area-ha", type=float, default=0.0,
                        help="drop polygons smaller than this (cleans up speckle noise)")
    parser.add_argument("--open", action="store_true", help="open the map in your browser")
    args = parser.parse_args()

    if args.mode == "sar" and args.raster == str(SAMPLE_DIR / "optical.tif"):
        args.raster = str(SAMPLE_DIR / "sar.tif")

    run(args.raster, args.mode, args.index, args.threshold, args.regime, args.k_cal,
        args.min_area_ha, args.open)


if __name__ == "__main__":
    main()
