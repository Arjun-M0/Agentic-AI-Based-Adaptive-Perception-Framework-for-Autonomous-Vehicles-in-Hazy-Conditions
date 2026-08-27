import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Smoke test to ensure the backbone architecture can be imported."""
    from src.core.context import FrameContext
    from src.agents.decision_agent import AdaptiveDecisionAgent
    assert True
