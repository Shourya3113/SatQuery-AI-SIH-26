"""
SatQuery AI - Core Data Models and Universal Schemas
Owner: Achintya (Backend Lead) & Peter (Team Leader)
Strictly adheres to ISRO SIH26167 functional requirements.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    OPTICAL_RGB = "OPTICAL_RGB"
    OPTICAL_MULTISPECTRAL = "OPTICAL_MULTISPECTRAL"
    SAR_C_BAND = "SAR_C_BAND"
    SAR_L_BAND = "SAR_L_BAND"
    BENCHMARK_IMAGE = "BENCHMARK_IMAGE"


class TaskCategory(str, Enum):
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    SINGLE_IMAGE_CAPTIONING = "SINGLE_IMAGE_CAPTIONING"
    TEXT_GUIDED_GROUNDING = "TEXT_GUIDED_GROUNDING"
    BI_TEMPORAL_CHANGE_DETECTION = "BI_TEMPORAL_CHANGE_DETECTION"
    CROSS_MODAL_JOINT_ANALYSIS = "CROSS_MODAL_JOINT_ANALYSIS"
    UNKNOWN = "UNKNOWN"


class InputImageMetadata(BaseModel):
    filename: str
    format: str  # GeoTIFF, TIFF, PNG, JPEG
    modality: ModalityType
    crs: Optional[str] = None
    width: int
    height: int
    bands: int
    spatial_resolution_m: Optional[float] = 10.0
    bounding_box: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
    co_registered: bool = False


class ToolExecutionStep(BaseModel):
    step_number: int
    tool_name: str
    parameters: Dict[str, Any]
    status: str = "SUCCESS"
    duration_ms: float


class VectorFeature(BaseModel):
    layer_name: str
    feature_type: str = "FeatureCollection"
    feature_count: int
    geojson: Dict[str, Any]
    metrics: Dict[str, Any] = Field(default_factory=dict)  # area_hectares, change_percentage


class XAIExplanation(BaseModel):
    method: str
    modality_attribution: Optional[Dict[str, float]] = None
    spectral_sensitivity: Optional[Dict[str, float]] = None
    physics_rationale: Optional[Dict[str, Any]] = None
    heatmap_overlay_base64: Optional[str] = None
    confidence: Optional[float] = None
    faithfulness_metrics: Optional[Dict[str, float]] = None
    runtime_ms: Optional[float] = None
    hardware_tier: Optional[str] = None
    limitations: Optional[List[str]] = Field(default_factory=list)
    summary: str


class AuditableExecutionTrace(BaseModel):
    trace_id: str
    timestamp: str
    user_query: str
    input_audit: Dict[str, Any]
    orchestration: Dict[str, Any]
    results: Dict[str, Any]
    # Telemetry and frontend compatibility fields
    selected_tool: Optional[str] = None
    reasoning: Optional[str] = None
    inputs: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    trace_id: str
    query: str
    task_category: TaskCategory
    text_response: str
    vector_layers: List[VectorFeature] = Field(default_factory=list)
    confidence_score: float
    execution_trace: AuditableExecutionTrace
    validation: Optional[Dict[str, Any]] = None
    xai_explanation: Optional[XAIExplanation] = None
