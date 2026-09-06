import os
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv, set_key
from app.services.data_validator import validate_and_extract_metadata
from app.services.agent import get_agent_router_trace

router = APIRouter()

# In-memory store for stats and history
stats_store = {
    "total_processed": 0,
    "success_count": 0,
    "total_time_ms": 0,
    "recent_queries": []
}

class SettingsUpdate(BaseModel):
    google_api_key: str

@router.get("/stats")
async def get_stats():
    success_rate = (stats_store["success_count"] / stats_store["total_processed"] * 100) if stats_store["total_processed"] > 0 else 0
    avg_time_ms = (stats_store["total_time_ms"] / stats_store["total_processed"]) if stats_store["total_processed"] > 0 else 0
    return {
        "total_processed": stats_store["total_processed"],
        "success_rate": f"{success_rate:.1f}%",
        "avg_time": f"{(avg_time_ms / 1000):.1f}s",
        "recent_queries": stats_store["recent_queries"]
    }

@router.get("/settings")
async def get_settings():
    load_dotenv(override=True)
    key = os.environ.get("GOOGLE_API_KEY", "")
    # Return masked key
    masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else ""
    return {"google_api_key_masked": masked, "has_key": bool(key)}

@router.post("/settings")
async def update_settings(settings: SettingsUpdate):
    dotenv_path = os.path.join(os.getcwd(), ".env")
    set_key(dotenv_path, "GOOGLE_API_KEY", settings.google_api_key)
    load_dotenv(override=True)
    return {"message": "Settings updated successfully"}

@router.post("/query")
async def process_query(
    query: str = Form(...),
    files: List[UploadFile] = File(...)
):
    start_time = time.time()
    stats_store["total_processed"] += 1
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    file_data = []
    for file in files:
        content = await file.read()
        file_data.append((file.filename, content))
        
    validation_result = validate_and_extract_metadata(file_data)
    
    if not validation_result.is_valid:
        raise HTTPException(status_code=400, detail=validation_result.message)
        
    trace = get_agent_router_trace(query=query, validation_result=validation_result)
    
    # Update stats
    stats_store["success_count"] += 1 if trace.get("selected_tool") != "error" else 0
    stats_store["total_time_ms"] += (time.time() - start_time) * 1000
    
    # Store query in history (keep last 10)
    stats_store["recent_queries"].insert(0, query)
    stats_store["recent_queries"] = stats_store["recent_queries"][:10]
    
    return {
        "query": query,
        "validation": validation_result.model_dump(),
        "execution_trace": trace
    }
