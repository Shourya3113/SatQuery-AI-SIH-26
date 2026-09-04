"""
SatQuery AI - Base Specialist Tool Class
Owner: Chhavi (AI & Deep Learning Lead)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import time


class BaseSpecialistTool(ABC):
    """
    Base class for SatQuery AI specialist tools.
    Enforces standardized execution traces, timing, and parameter bounds.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executes tool logic.
        Returns: (output_dict, step_telemetry_dict)
        """
        pass

    def run_with_telemetry(self, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        start_time = time.time()
        output, telemetry = self.execute(inputs, parameters)
        duration_ms = round((time.time() - start_time) * 1000.0, 2)

        telemetry["tool_name"] = self.name
        telemetry["duration_ms"] = duration_ms
        telemetry["parameters"] = parameters
        return output, telemetry
