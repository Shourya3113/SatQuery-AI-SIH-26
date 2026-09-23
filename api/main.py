"""
SatQuery AI - FastAPI REST Gateway
Owner: Achintya (Backend Lead) & Vinayak (Secondary Backend)
Integrated with Peter's Agentic Task Orchestrator & Misha's Geospatial Pipeline.
"""

import os
import time
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv, set_key

from core.config import UPLOADS_DIR, OUTPUTS_DIR
from core.schemas import QueryResponse, TaskCategory
from services.orchestrator import AgenticTaskRouter
from services.report_generator import generate_report

# In-memory stores for traces, full responses, and operational stats
execution_traces: Dict[str, Dict[str, Any]] = {}
stored_responses: Dict[str, QueryResponse] = {}
stats_store: Dict[str, Any] = {
    "total_processed": 0,
    "success_count": 0,
    "total_time_ms": 0.0,
    "recent_queries": []
}


class SettingsUpdate(BaseModel):
    google_api_key: Optional[str] = None
    cesium_ion_token: Optional[str] = None


# Instantiate central Agentic Task Orchestrator
orchestrator = AgenticTaskRouter()

app = FastAPI(
    title="SatQuery AI REST Gateway",
    description="Agentic Multimodal Remote Sensing Intelligence Platform (ISRO SIH26167)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {
        "status": "HEALTHY",
        "service": "SatQuery AI Gateway",
        "isro_problem_id": "SIH26167",
        "orchestrator_status": "ACTIVE"
    }


@app.get("/api/stats")
async def get_stats():
    total = stats_store["total_processed"]
    success = stats_store["success_count"]
    time_ms = stats_store["total_time_ms"]
    success_rate = (success / total * 100.0) if total > 0 else 100.0
    avg_time_s = (time_ms / total / 1000.0) if total > 0 else 0.0
    return {
        "total_processed": total,
        "success_rate": f"{success_rate:.1f}%",
        "avg_time": f"{avg_time_s:.1f}s",
        "recent_queries": stats_store["recent_queries"]
    }


@app.get("/api/settings")
async def get_settings():
    load_dotenv(override=True)
    key = os.environ.get("GOOGLE_API_KEY", "")
    cesium_token = os.environ.get("CESIUM_ION_ACCESS_TOKEN", "")
    masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else ""
    cesium_masked = f"{cesium_token[:6]}...{cesium_token[-4:]}" if len(cesium_token) > 10 else ""
    return {
        "google_api_key_masked": masked,
        "has_key": bool(key),
        "cesium_ion_token_masked": cesium_masked,
        "has_cesium_key": bool(cesium_token),
    }


@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if settings.google_api_key is not None:
        try:
            set_key(dotenv_path, "GOOGLE_API_KEY", settings.google_api_key)
        except Exception:
            with open(dotenv_path, "a") as f:
                f.write(f"\nGOOGLE_API_KEY={settings.google_api_key}\n")
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    if settings.cesium_ion_token is not None:
        try:
            set_key(dotenv_path, "CESIUM_ION_ACCESS_TOKEN", settings.cesium_ion_token)
        except Exception:
            with open(dotenv_path, "a") as f:
                f.write(f"\nCESIUM_ION_ACCESS_TOKEN={settings.cesium_ion_token}\n")
        os.environ["CESIUM_ION_ACCESS_TOKEN"] = settings.cesium_ion_token

    load_dotenv(override=True)
    return {"message": "Settings updated successfully"}


@app.get("/api/samples")
async def list_sample_imagery():
    """Returns catalog of pre-packaged synthetic satellite imagery for instant demos."""
    return [
        {
            "id": "water_grounding",
            "title": "Sentinel-2 Optical (Water & Agriculture)",
            "description": "4-band multispectral scene (10m GSD) over water bodies and farmland.",
            "suggested_query": "Highlight and segment the water body in this image",
            "files": ["optical.tif"]
        },
        {
            "id": "bitemporal_change",
            "title": "Bi-Temporal Pair (Monsoon Flood Inundation)",
            "description": "Co-registered T1 baseline vs T2 post-flood acquisition.",
            "suggested_query": "What changed between these two temporal acquisitions?",
            "files": ["bitemporal_t1.tif", "bitemporal_t2.tif"]
        },
        {
            "id": "optical_sar_fusion",
            "title": "Optical RGB + SAR C-Band Microwave Pair",
            "description": "Cross-modal pair combining optical reflectance with cloud-penetrating SAR backscatter.",
            "suggested_query": "Use optical and SAR together to detect built-up and water covered regions",
            "files": ["optical.tif", "sar.tif"]
        }
    ]


@app.get("/api/samples/{filename}")
async def get_sample_file(filename: str):
    """Serves sample satellite imagery files directly to the client."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid path traversal.")
    file_path = Path("data/samples") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample file not found.")
    media_type = "image/tiff" if filename.endswith(".tif") else "image/png"
    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)


@app.get("/api/samples/preview/{filename}")
async def get_sample_preview_file(filename: str):
    """Serves sample RGB preview images directly to the client."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid path traversal.")
    file_path = Path("data/samples") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Preview file not found.")
    return FileResponse(path=str(file_path), media_type="image/png")


@app.get("/api/benchmarks/preview/{dataset}/{filename}")
async def get_benchmark_preview_file(dataset: str, filename: str):
    """Serves benchmark dataset preview images."""
    if ".." in dataset or ".." in filename or "/" in dataset or "\\" in dataset:
        raise HTTPException(status_code=400, detail="Invalid path traversal.")
    file_path = Path("data/benchmarks") / dataset / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Preview file not found.")
    media_type = "image/png" if filename.endswith(".png") else "image/tiff"
    return FileResponse(path=str(file_path), media_type=media_type)


@app.get("/api/benchmarks/gallery")
async def get_benchmark_gallery():
    """
    Returns gallery metadata and preview URLs for all 4 SIH26167 benchmark datasets:
    BigEarthNet-MM, VRSBench, RSVQA, and CDVQA.
    """
    return [
        {
            "id": "bigearthnet",
            "title": "BigEarthNet-MM: Optical-SAR Multimodal Fusion",
            "dataset": "BigEarthNet-MM",
            "description": "Co-registered Sentinel-2 4-band optical (10m GSD) and Sentinel-1 C-band SAR microwave backscatter (VV polarisation).",
            "spatial_crs": "EPSG:32633 (UTM Zone 33N)",
            "spatial_resolution": "10.0 m/px",
            "modalities": ["Sentinel-2 Optical (4-Band)", "Sentinel-1 SAR C-Band (VV)"],
            "images": [
                {
                    "name": "Sentinel-2 Optical RGB Composite",
                    "file": "s2_patch.tif",
                    "preview_url": "/api/benchmarks/preview/bigearthnet/s2_patch_preview.png",
                    "bands": "B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)",
                    "type": "Multispectral Optical"
                },
                {
                    "name": "Sentinel-1 SAR Backscatter Intensity",
                    "file": "s1_patch.tif",
                    "preview_url": "/api/benchmarks/preview/bigearthnet/s1_patch_preview.png",
                    "bands": "VV Polarisation (Microwave C-Band)",
                    "type": "SAR Radar Backscatter"
                }
            ]
        },
        {
            "id": "cdvqa",
            "title": "CDVQA: Bi-Temporal Flood Inundation & Change Detection",
            "dataset": "CDVQA",
            "description": "Bi-temporal satellite observations capturing pre-flood baseline (T1) versus post-monsoon water body expansion (T2).",
            "spatial_crs": "EPSG:32633 (UTM Zone 33N)",
            "spatial_resolution": "10.0 m/px",
            "modalities": ["Bi-Temporal Sentinel-2 Multispectral"],
            "images": [
                {
                    "name": "T1 Pre-Flood Baseline",
                    "file": "cdvqa_t1.tif",
                    "preview_url": "/api/benchmarks/preview/cdvqa/cdvqa_t1_preview.png",
                    "acquisition": "2023-01-10 Baseline",
                    "type": "Pre-Event Optical"
                },
                {
                    "name": "T2 Post-Flood Inundation",
                    "file": "cdvqa_t2.tif",
                    "preview_url": "/api/benchmarks/preview/cdvqa/cdvqa_t2_preview.png",
                    "acquisition": "2023-08-20 Post-Event",
                    "type": "Post-Event Optical"
                },
                {
                    "name": "Ground Truth Inundation Mask",
                    "file": "gt_change_mask.npy",
                    "preview_url": "/api/benchmarks/preview/cdvqa/gt_change_mask_preview.png",
                    "metrics": "15.2% verified flood inundation mask (Red highlight)",
                    "type": "Change Mask"
                }
            ]
        },
        {
            "id": "vrsbench",
            "title": "VRSBench: Visual Spatial Grounding & Delineation",
            "dataset": "VRSBench",
            "description": "High-resolution remote sensing scene with natural-language spatial grounding annotations and pixel segmentation mask.",
            "spatial_crs": "EPSG:32633 (Projected Coordinates)",
            "spatial_resolution": "0.5 m/px High-Resolution",
            "modalities": ["Aerial / VHR Optical"],
            "images": [
                {
                    "name": "High-Res Aerial Scene",
                    "file": "vrsbench_scene_001.png",
                    "preview_url": "/api/benchmarks/preview/vrsbench/vrsbench_scene_001.png",
                    "prompt": "Segment and ground the target building complex",
                    "type": "VHR Optical"
                },
                {
                    "name": "Ground Truth Target Mask",
                    "file": "gt_mask_001.npy",
                    "preview_url": "/api/benchmarks/preview/vrsbench/gt_mask_001_preview.png",
                    "metrics": "Target baseline: >= 0.65 mIoU",
                    "type": "Delineation Mask"
                }
            ]
        },
        {
            "id": "rsvqa",
            "title": "RSVQA: Remote Sensing Visual Question Answering",
            "dataset": "RSVQA",
            "description": "Multispectral satellite observation evaluated across semantic presence, counting, and scene description questions.",
            "spatial_crs": "EPSG:32633 (UTM Zone 33N)",
            "spatial_resolution": "10.0 m/px",
            "modalities": ["Sentinel-2 Optical Multispectral"],
            "images": [
                {
                    "name": "RSVQA Multispectral Optical Scene",
                    "file": "rsvqa_optical.tif",
                    "preview_url": "/api/benchmarks/preview/rsvqa/rsvqa_optical_preview.png",
                    "eval": "3 QA pairs evaluated, Mean BLEU-2: 0.5713",
                    "type": "Multispectral Tile"
                }
            ]
        }
    ]


@app.get("/api/benchmarks/evaluate")
async def run_benchmark_evaluation():
    """
    Executes the official SIH26167 public benchmark evaluation harness across:
    - BigEarthNet-MM
    - VRSBench
    - RSVQA
    - CDVQA
    Returns the normalized composite scorecard and metrics.
    """
    from scripts.run_benchmarks import evaluate_benchmarks
    try:
        scorecard = evaluate_benchmarks()
        return scorecard
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark evaluation failed: {str(e)}")


@app.get("/api/benchmarks/faithfulness")
async def run_faithfulness_benchmark_endpoint(force_refresh: bool = False):
    """
    Executes or retrieves the RS-XAI Scientific Faithfulness Benchmark:
    - Multispectral Permutation Feature Importance (PFI) across Sentinel-2 (B02, B03, B04, B08) & SAR C-band.
    - Area Over Perturbation Curve (AOPC) deletion-insertion tests (MoRF vs Random vs LeRF).
    - Remote sensing physics consistency checks (NDWI & NDVI).
    """
    import json
    scorecard_path = Path(__file__).resolve().parent.parent / "reports" / "spectral_pfi_scorecard.json"

    if scorecard_path.exists() and not force_refresh:
        try:
            return json.loads(scorecard_path.read_text())
        except Exception:
            pass

    from scripts.spectral_pfi_benchmark import run_full_benchmark
    try:
        scorecard = run_full_benchmark()
        scorecard_path.parent.mkdir(parents=True, exist_ok=True)
        scorecard_path.write_text(json.dumps(scorecard, indent=2))
        return scorecard
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scientific faithfulness benchmark failed: {str(e)}")



@app.post("/api/upload")
async def upload_rasters(files: List[UploadFile] = File(...)):
    """
    Ingests GeoTIFF, TIFF, PNG, or JPEG remote sensing imagery.
    """
    saved_files = []
    for f in files:
        safe_filename = Path(f.filename).name
        target_path = UPLOADS_DIR / safe_filename
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_files.append({
            "filename": safe_filename,
            "path": str(target_path),
            "size_bytes": target_path.stat().st_size
        })
    return {"uploaded_files": saved_files, "status": "SUCCESS"}


@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    confidence_threshold: Optional[float] = Form(None),
    change_threshold: Optional[float] = Form(None),
    speckle_filter_kernel: Optional[int] = Form(None),
    max_tokens: Optional[int] = Form(None),
    include_xai: Optional[bool] = Form(False)
):
    """
    Receives remote sensing imagery (single, bi-temporal pair, or cross-modal optical+SAR pair)
    and executes the central Agentic Task Orchestrator.
    Returns complete QueryResponse with verifiable AuditableExecutionTrace.
    """
    start_time = time.time()
    uploaded_files: List[UploadFile] = []
    if image is not None:
        uploaded_files.append(image)
    if files is not None:
        uploaded_files.extend(files)

    if not uploaded_files:
        raise HTTPException(
            status_code=400,
            detail="No image uploaded. Provide either 'image' or 'files'."
        )

    saved_paths: List[Path] = []
    for f in uploaded_files:
        safe_filename = Path(f.filename).name
        target_path = UPLOADS_DIR / safe_filename
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_paths.append(target_path)

    # Assemble raw parameter dictionary
    raw_params: Dict[str, Any] = {}
    if confidence_threshold is not None:
        raw_params["confidence_threshold"] = confidence_threshold
    if change_threshold is not None:
        raw_params["change_threshold"] = change_threshold
    if speckle_filter_kernel is not None:
        raw_params["speckle_filter_kernel"] = speckle_filter_kernel
    if max_tokens is not None:
        raw_params["max_tokens"] = max_tokens
    if include_xai is not None:
        raw_params["include_xai"] = include_xai

    # Execute Peter's Agentic Task Router
    try:
        response = orchestrator.process_query(
            query=query,
            file_paths=saved_paths,
            raw_params=raw_params
        )
    except Exception as e:
        stats_store["total_processed"] += 1
        raise HTTPException(
            status_code=500,
            detail=f"Agentic orchestration failure: {str(e)}"
        )

    elapsed_ms = (time.time() - start_time) * 1000.0
    stats_store["total_processed"] += 1
    stats_store["success_count"] += 1
    stats_store["total_time_ms"] += elapsed_ms
    stats_store["recent_queries"].insert(0, query)
    stats_store["recent_queries"] = stats_store["recent_queries"][:10]

    # Persist trace and response
    execution_traces[response.trace_id] = response.execution_trace.model_dump()
    stored_responses[response.trace_id] = response

    return response


@app.get("/api/trace/{trace_id}")
async def get_execution_trace(trace_id: str):
    """
    Fetches the stored execution trace for a given trace ID.
    """
    if ".." in trace_id or "/" in trace_id or "\\" in trace_id:
        raise HTTPException(status_code=400, detail="Invalid trace ID.")
    trace = execution_traces.get(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"Execution trace '{trace_id}' not found."
        )
    return trace


@app.post("/api/export-report/{trace_id}")
async def export_report(trace_id: str):
    """
    Generates and returns an Intelligence Dossier PDF
    for a previously executed query using the stored QueryResponse.
    """
    if ".." in trace_id or "/" in trace_id or "\\" in trace_id:
        raise HTTPException(status_code=400, detail="Invalid trace ID.")
    query_response = stored_responses.get(trace_id)

    if query_response is None:
        trace = execution_traces.get(trace_id)
        if trace is None:
            raise HTTPException(
                status_code=404,
                detail=f"Execution trace '{trace_id}' not found."
            )

        # Reconstruct QueryResponse from trace
        query_response = QueryResponse(
            trace_id=trace_id,
            query=trace["user_query"],
            task_category=TaskCategory(trace["orchestration"]["selected_task"]),
            text_response=trace["results"].get("textual_summary", "Analysis completed."),
            confidence_score=trace["results"].get("overall_confidence", 0.90),
            execution_trace=trace
        )

    # Save PDF to outputs directory
    safe_trace_id = Path(trace_id).name
    report_path = str(OUTPUTS_DIR / f"{safe_trace_id}_report.pdf")

    # Generate the PDF dossier
    generate_report(query_response, report_path)

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=f"{safe_trace_id}_report.pdf"
    )