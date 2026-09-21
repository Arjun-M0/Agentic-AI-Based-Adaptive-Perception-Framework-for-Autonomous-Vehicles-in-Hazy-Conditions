import pytest
from src.agents.hardware_agent import HardwareAwarenessAgent
from src.core.context import HardwareState

def test_hardware_agent_returns_correct_type():
    """Verify that the agent returns the exact typed dataclass defined in our contracts."""
    agent = HardwareAwarenessAgent()
    state = agent.analyze()
    assert isinstance(state, HardwareState), "Agent should return a HardwareState instance"

def test_hardware_metrics_within_valid_ranges():
    """Verify that CPU and RAM percentages are realistic floats (0.0 to 100.0)."""
    agent = HardwareAwarenessAgent()
    state = agent.analyze()
    
    # Check CPU
    assert state.cpu_utilization is not None
    assert 0.0 <= state.cpu_utilization <= 100.0, "CPU utilization must be between 0 and 100"
    
    # Check RAM
    assert state.ram_utilization is not None
    assert 0.0 <= state.ram_utilization <= 100.0, "RAM utilization must be between 0 and 100"

def test_gpu_fallback_graceful():
    """
    Verify that if a GPU is not available, the fallback logic executes 
    gracefully and the dependent fields safely remain None.
    """
    agent = HardwareAwarenessAgent()
    state = agent.analyze()
    
    if not state.gpu_available:
        assert state.gpu_utilization is None, "GPU utilization should be None if GPU is unavailable"
        assert state.gpu_memory_available is None, "GPU memory should be None if GPU is unavailable"
