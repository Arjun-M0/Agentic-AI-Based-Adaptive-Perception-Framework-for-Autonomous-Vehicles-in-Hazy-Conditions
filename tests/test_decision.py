import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.decision_agent import AdaptiveDecisionAgent, Decision
from src.core.context import EnvironmentState, HardwareState

@pytest.fixture
def agent():
    return AdaptiveDecisionAgent()

def test_decision_with_empty_knowledge(agent):
    env = EnvironmentState(haze_score=0.2, visibility_score=0.8)
    hw = HardwareState(gpu_available=False)
    
    decision = agent.select_strategy(environment_state=env, hardware_state=hw, historical_knowledge=[])
    
    assert isinstance(decision, Decision)
    assert decision.selected_strategy in agent.strategies
    assert decision.constraints_satisfied is True
    assert decision.evidence_strength == 0.0
    assert "No historical evidence was available" in decision.explanation

def test_hard_constraints_gpu(agent):
    env = EnvironmentState(haze_score=0.9, visibility_score=0.1)
    hw = HardwareState(gpu_available=False)
    
    decision = agent.select_strategy(environment_state=env, hardware_state=hw)
    
    assert decision.selected_strategy != "DehazeFormer"
    assert decision.strategy_scores["DehazeFormer"] == 0.0
    assert "DehazeFormer" in decision.explanation

def test_historical_evidence_effect(agent):
    env = EnvironmentState(haze_score=0.5, visibility_score=0.5)
    hw = HardwareState(gpu_available=True)
    
    history = [
        {"strategy": "DEA-Net", "outcome": {"success": True}, "environment": {"haze_score": 0.5}},
        {"strategy": "DEA-Net", "outcome": {"success": True}, "environment": {"haze_score": 0.6}},
        {"strategy": "Bypass", "outcome": {"success": False}, "environment": {"haze_score": 0.5}}
    ]
    
    decision = agent.select_strategy(environment_state=env, hardware_state=hw, historical_knowledge=history)
    assert decision.selected_strategy == "DEA-Net"
    assert decision.evidence_strength > 0.0

def test_performance_constraints_respected(agent):
    env = EnvironmentState(haze_score=0.5)
    hw = HardwareState(gpu_available=True)
    requirements = {"max_latency": 100, "target_fps": 30}
    
    performance_data = {
        "DEA-Net": {"latency": 150, "fps": 40},
        "DehazeFormer": {"latency": 80, "fps": 20},
        "Bypass": {"latency": 10, "fps": 60}
    }
    
    decision = agent.select_strategy(
        environment_state=env, 
        hardware_state=hw, 
        requirements=requirements,
        performance_data=performance_data
    )
    
    assert decision.selected_strategy == "Bypass"
    assert decision.strategy_scores["DEA-Net"] == 0.0
    assert decision.strategy_scores["DehazeFormer"] == 0.0
    assert "Latency" in decision.explanation or "FPS" in decision.explanation

def test_unevaluated_constraints_reported(agent):
    env = EnvironmentState(haze_score=0.5)
    hw = HardwareState(gpu_available=True)
    requirements = {"max_latency": 100, "target_fps": 30}
    
    decision = agent.select_strategy(
        environment_state=env, 
        hardware_state=hw, 
        requirements=requirements
    )
    
    assert "Constraints unverified due to missing data: latency, fps" in decision.explanation

def test_hardware_state_effect(agent):
    env = EnvironmentState(haze_score=0.5)
    hw_no_gpu = HardwareState(gpu_available=False)
    hw_with_gpu = HardwareState(gpu_available=True)
    
    decision_no_gpu = agent.select_strategy(environment_state=env, hardware_state=hw_no_gpu)
    decision_with_gpu = agent.select_strategy(environment_state=env, hardware_state=hw_with_gpu)
    
    assert decision_no_gpu.strategy_scores["DEA-Net"] < decision_with_gpu.strategy_scores["DEA-Net"]
    assert decision_no_gpu.selected_strategy != "DehazeFormer"

def test_deterministic_output(agent):
    env = EnvironmentState(haze_score=0.5)
    hw = HardwareState(gpu_available=True)
    
    decision1 = agent.select_strategy(environment_state=env, hardware_state=hw)
    decision2 = agent.select_strategy(environment_state=env, hardware_state=hw)
    
    assert decision1.selected_strategy == decision2.selected_strategy
    assert decision1.strategy_scores == decision2.strategy_scores
