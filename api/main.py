"""
SatQuery AI - FastAPI REST Gateway
Owner: Achintya (Backend Lead) & Vinayak (Secondary Backend)
"""
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
from pathlib import Path

from core.config import UPLOADS_DIR
from core.schemas import QueryResponse
from fastapi.responses import FileResponse
from services.report_generator import generate_report

# Temporary storage for execution traces
execution_traces = {}

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
        "isro_problem_id": "SIH26167"
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
    image: UploadFile = File(...)
):
    """
    Receives a satellite image and a natural-language query.
    Currently returns a mock response for backend testing.
    """

    # Save the uploaded image
    target_path = UPLOADS_DIR / image.filename

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Generate a unique trace ID
    trace_id = f"satquery-exec-{uuid.uuid4().hex[:8]}"

    # Create the execution trace
    trace = {
    "trace_id": trace_id,
    "timestamp": "2026-09-04T21:00:00Z",
    "user_query": query,
    "input_audit": {
        "filename": image.filename,
        "content_type": image.content_type
    },
    "orchestration": {
        "selected_task": "SINGLE_IMAGE_VQA",
        "pipeline_steps": [
            {
                "step_number": 1,
                "tool_name": "image_loader",
                "parameters": {
                    "filename": image.filename
                },
                "status": "SUCCESS",
                "duration_ms": 100.0
            }
        ]
    },
    "results": {}
}

    # Store the trace in memory
    execution_traces[trace_id] = trace

    # Return the query response
    return QueryResponse(
        trace_id=trace_id,
        query=query,
        task_category="SINGLE_IMAGE_VQA",
        text_response="Mock response: image received successfully.",
        confidence_score=0.90,
        execution_trace=trace
    )

@app.get("/api/trace/{trace_id}")
async def get_execution_trace(trace_id: str):
    """
    Fetches the stored execution trace for a given trace ID.
    """

    trace = execution_traces.get(trace_id)

    if trace is None:
        raise HTTPException(
            status_code=404,
            detail="Trace not found"
        )

    return trace

@app.post("/api/export-report/{trace_id}")
async def export_report(trace_id: str):
    """
    Generates and returns an Intelligence Dossier PDF
    for a previously executed query.
    """

    # Find the stored execution trace
    trace = execution_traces.get(trace_id)

    if trace is None:
        raise HTTPException(
            status_code=404,
            detail="Trace not found"
        )

    # Reconstruct the QueryResponse
    query_response = QueryResponse(
        trace_id=trace_id,
        query=trace["user_query"],
        task_category=trace["orchestration"]["selected_task"],
        text_response="Mock response: image received successfully.",
        confidence_score=0.90,
        execution_trace=trace
    )

    # Create the output PDF path
    report_path = str(UPLOADS_DIR / f"{trace_id}_report.pdf")

    # Generate the PDF
    generate_report(
        query_response,
        report_path
    )

    # Return the generated PDF
    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=f"{trace_id}_report.pdf"
    )