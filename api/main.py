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
    google_api_key: str


# Instantiate central Agentic Task Orchestrator
orchestrator = AgenticTaskRouter()

app = FastAPI(
    title="SatQuery AI REST Gateway",
    description="Agentic Multimodal Remote Sensing Intelligence Platform (ISRO SIH26167)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else ""
    return {
        "google_api_key_masked": masked,
        "has_key": bool(key)
    }


@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    dotenv_path = os.path.join(os.getcwd(), ".env")
    try:
        set_key(dotenv_path, "GOOGLE_API_KEY", settings.google_api_key)
    except Exception:
        with open(dotenv_path, "a") as f:
            f.write(f"\nGOOGLE_API_KEY={settings.google_api_key}\n")
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
    file_path = Path("data/samples") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample file not found.")
    media_type = "image/tiff" if filename.endswith(".tif") else "image/png"
    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)


@app.post("/api/upload")
async def upload_rasters(files: List[UploadFile] = File(...)):
    """
    Ingests GeoTIFF, TIFF, PNG, or JPEG remote sensing imagery.
    """
    saved_files = []
    for f in files:
        target_path = UPLOADS_DIR / f.filename
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_files.append({
            "filename": f.filename,
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
    max_tokens: Optional[int] = Form(None)
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
        target_path = UPLOADS_DIR / f.filename
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
    report_path = str(OUTPUTS_DIR / f"{trace_id}_report.pdf")

    # Generate the PDF dossier
    generate_report(query_response, report_path)

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=f"{trace_id}_report.pdf"
    )