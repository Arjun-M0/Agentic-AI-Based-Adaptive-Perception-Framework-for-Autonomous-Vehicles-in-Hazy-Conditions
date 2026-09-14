import pytest
import numpy as np
from src.agents.environment_agent import EnvironmentUnderstandingAgent
from src.core.context import EnvironmentState

def test_environment_agent_returns_correct_type():
    """Verify the agent returns the exact typed dataclass defined in our contracts."""
    agent = EnvironmentUnderstandingAgent()
    # Create a completely black 100x100 dummy image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    state = agent.analyze(dummy_image)
    assert isinstance(state, EnvironmentState), "Agent should return an EnvironmentState instance"

def test_environment_metrics_valid_ranges():
    """Verify that all scores are floats within the standardized 0.0 to 1.0 range."""
    agent = EnvironmentUnderstandingAgent()
    # Create a random noisy image
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    state = agent.analyze(dummy_image)
    
    assert state.illumination_score is not None
    assert 0.0 <= state.illumination_score <= 1.0, "Illumination score out of bounds"
    
    assert state.contrast_score is not None
    assert 0.0 <= state.contrast_score <= 1.0, "Contrast score out of bounds"
    
    assert state.haze_score is not None
    assert 0.0 <= state.haze_score <= 1.0, "Haze score out of bounds"
    
    assert state.visibility_score is not None
    assert 0.0 <= state.visibility_score <= 1.0, "Visibility score out of bounds"

def test_empty_image_handling_graceful():
    """Verify the agent handles corrupted or empty images gracefully without crashing."""
    agent = EnvironmentUnderstandingAgent()
    
    # Pass None
    state_none = agent.analyze(None)
    assert state_none.haze_score is None, "Score should be None for empty image"
    
    # Pass empty array
    state_empty = agent.analyze(np.array([]))
    assert state_empty.visibility_score is None, "Score should be None for empty image"
