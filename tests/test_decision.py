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
    assert "Bypass" in decision.strategy_scores
    assert "DEA-Net" in decision.strategy_scores
    assert "DehazeFormer" in decision.strategy_scores
    assert decision.evidence_strength == 0.0
    assert "No historical experiences were available" in decision.explanation

def test_hard_constraints(agent):
    env = EnvironmentState(haze_score=0.9, visibility_score=0.1)
    hw = HardwareState(gpu_available=False)
    
    # DehazeFormer requires GPU in the current mock logic
    decision = agent.select_strategy(environment_state=env, hardware_state=hw)
    
    # It should not select DehazeFormer even with high haze, because GPU is False
    assert decision.selected_strategy != "DehazeFormer"
    assert decision.strategy_scores["DehazeFormer"] == 0.0

def test_historical_evidence_effect(agent):
    env = EnvironmentState(haze_score=0.5, visibility_score=0.5)
    hw = HardwareState(gpu_available=True)
    
    # Provide history where DEA-Net was very successful
    history = [
        {"strategy": "DEA-Net", "outcome": {"success": True}},
        {"strategy": "DEA-Net", "outcome": {"success": True}},
        {"strategy": "Bypass", "outcome": {"success": False}}
    ]
    
    decision = agent.select_strategy(environment_state=env, hardware_state=hw, historical_knowledge=history)
    assert decision.selected_strategy == "DEA-Net"
    assert decision.evidence_strength > 0.0

def test_environmental_state_effect(agent):
    env_low_haze = EnvironmentState(haze_score=0.1)
    env_high_haze = EnvironmentState(haze_score=0.9)
    hw = HardwareState(gpu_available=True)
    
    decision_low = agent.select_strategy(environment_state=env_low_haze, hardware_state=hw)
    decision_high = agent.select_strategy(environment_state=env_high_haze, hardware_state=hw)
    
    assert decision_low.selected_strategy == "Bypass"
    assert decision_high.selected_strategy == "DehazeFormer"

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

def test_explanation_generation(agent):
    env = EnvironmentState(haze_score=0.9)
    hw = HardwareState(gpu_available=False)
    history = [{"strategy": "Bypass", "outcome": {"success": True}}]
    
    decision = agent.select_strategy(environment_state=env, hardware_state=hw, historical_knowledge=history)
    
    assert "DehazeFormer marked infeasible" in decision.explanation
    assert decision.selected_strategy in decision.explanation
