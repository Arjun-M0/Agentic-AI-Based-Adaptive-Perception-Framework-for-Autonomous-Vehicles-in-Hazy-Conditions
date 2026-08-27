from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class EnvironmentState:
    haze_score: Optional[float] = None
    visibility_score: Optional[float] = None
    illumination_score: Optional[float] = None
    contrast_score: Optional[float] = None

@dataclass
class HardwareState:
    gpu_available: bool = False
    gpu_utilization: Optional[float] = None
    gpu_memory_available: Optional[float] = None
    cpu_utilization: Optional[float] = None
    ram_utilization: Optional[float] = None
    target_fps: Optional[float] = None

@dataclass
class FrameContext:
    frame_id: int
    frame: Any
    environment_state: Optional[EnvironmentState] = None
    hardware_state: Optional[HardwareState] = None
    historical_knowledge: Optional[Dict] = None
    decision: Optional[str] = None
    restoration_result: Any = None
    detection_result: Any = None
    evaluation_result: Optional[Dict] = None
