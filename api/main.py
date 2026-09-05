"""
SatQuery AI - FastAPI REST Gateway
Owner: Achintya (Backend Lead) & Vinayak (Secondary Backend)
Integrated with Peter's Agentic Task Orchestrator & Misha's Geospatial Pipeline.
"""

import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.config import UPLOADS_DIR, OUTPUTS_DIR
from core.schemas import QueryResponse, TaskCategory
from services.orchestrator import AgenticTaskRouter
from services.report_generator import generate_report

# In-memory stores for traces and full responses
execution_traces: Dict[str, Dict[str, Any]] = {}
stored_responses: Dict[str, QueryResponse] = {}

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
        raise HTTPException(
            status_code=500,
            detail=f"Agentic orchestration failure: {str(e)}"
        )

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