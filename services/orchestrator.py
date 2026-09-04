"""
SatQuery AI - Agentic Task Orchestrator
Owner: Peter (Team Leader, Chief Architect & Integration Lead)
Strictly complies with ISRO SIH26167:
- Intent Classification
- Input Compatibility & Co-registration Verification
- Dynamic Tool Dispatcher
- Permitted Parameter Bounds Enforcement
- Auditable Execution Trace Synthesizer
"""

import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.schemas import (
    TaskCategory,
    ModalityType,
    ToolExecutionStep,
    VectorFeature,
    AuditableExecutionTrace,
    QueryResponse
)
from core.config import PERMITTED_PARAMETERS
from services.geospatial import GeospatialEngine


class AgenticTaskRouter:
    """
    Central Orchestrator for SatQuery AI.
    Routes queries to specialist engines and emits verifiable JSON execution traces.
    """

    def __init__(self, tool_registry: Optional[Dict[str, Any]] = None):
        self.tool_registry = tool_registry or {}

    def classify_task(self, query: str, input_count: int, modalities: List[ModalityType]) -> TaskCategory:
        """
        Classifies query intent against input configuration.
        """
        q = query.lower()

        if input_count >= 2 and any("sar" in m.value.lower() for m in modalities) and any("optical" in m.value.lower() for m in modalities):
            return TaskCategory.CROSS_MODAL_JOINT_ANALYSIS

        if ("together" in q or "both" in q or "sar and optical" in q) and input_count >= 2:
            return TaskCategory.CROSS_MODAL_JOINT_ANALYSIS

        if ("change" in q or "different dates" in q or "increased" in q or "between" in q) and input_count >= 2:
            return TaskCategory.BI_TEMPORAL_CHANGE_DETECTION

        if "highlight" in q or "segment" in q or "delineate" in q or "boundary" in q:
            return TaskCategory.TEXT_GUIDED_GROUNDING

        if "describe" in q or "caption" in q or "summary" in q or "overview" in q:
            return TaskCategory.SINGLE_IMAGE_CAPTIONING

        return TaskCategory.SINGLE_IMAGE_VQA

    def validate_and_bound_parameters(self, raw_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enforces strict parameter bounds per ISRO requirements.
        """
        bounded = {}
        raw = raw_params or {}

        for key, spec in PERMITTED_PARAMETERS.items():
            val = raw.get(key, spec.get("default"))
            if "min" in spec and "max" in spec:
                bounded[key] = max(spec["min"], min(spec["max"], float(val)))
            elif "allowed" in spec:
                bounded[key] = val if val in spec["allowed"] else spec["default"]
            else:
                bounded[key] = val
        return bounded

    def build_execution_trace(
        self,
        trace_id: str,
        user_query: str,
        input_audit: Dict[str, Any],
        selected_task: str,
        pipeline_steps: List[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> AuditableExecutionTrace:
        """
        Synthesizes the verifiable observable JSON execution trace.
        """
        return AuditableExecutionTrace(
            trace_id=trace_id,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            user_query=user_query,
            input_audit=input_audit,
            orchestration={
                "selected_task": selected_task,
                "pipeline_steps": pipeline_steps
            },
            results=results
        )
