import logging
import psutil
from typing import Optional

# We'll try to use pynvml for lightweight, device-agnostic NVIDIA GPU monitoring 
# without strictly requiring massive PyTorch overhead if it's not needed for inference here.
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from src.core.context import HardwareState
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class HardwareAwarenessAgent(BaseAgent):
    """
    Agent responsible for monitoring the system's hardware resources.
    Collects real-time metrics on CPU, RAM, and GPU to inform the Adaptive Decision Agent.
    """

    def __init__(self):
        super().__init__()
        self.nvml_initialized = False
        
        # Safely attempt to initialize NVML for GPU monitoring
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("NVML successfully initialized for GPU monitoring.")
            except Exception as e:
                logger.warning(f"pynvml is installed, but initialization failed (e.g., no NVIDIA GPU found): {e}")

    def analyze(self) -> HardwareState:
        """
        Analyzes the current computational environment.
        Returns a HardwareState with CPU, RAM, and GPU metrics.
        """
        state = HardwareState()
        
        # 1. CPU Metrics
        # interval=0.1 provides a quick, non-blocking reading of CPU usage over 100ms
        state.cpu_utilization = psutil.cpu_percent(interval=0.1)
        
        # 2. RAM Metrics
        vm = psutil.virtual_memory()
        state.ram_utilization = vm.percent
        
        # 3. GPU Metrics
        if self.nvml_initialized:
            try:
                # Get handle for the first GPU (index 0)
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU Utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                state.gpu_utilization = float(utilization.gpu)
                
                # GPU Memory
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # Convert free memory from bytes to MB
                state.gpu_memory_available = float(memory.free) / (1024 ** 2)
                state.gpu_available = True
                
            except Exception as e:
                logger.warning(f"Failed to read GPU metrics: {e}")
                state.gpu_available = False
        else:
            state.gpu_available = False

        return state
