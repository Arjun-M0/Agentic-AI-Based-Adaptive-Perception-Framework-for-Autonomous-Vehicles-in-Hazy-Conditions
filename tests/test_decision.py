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
    env = EnvironmentState(haze_score=0.2, visibility_score=0.8, illumination_score=0.5, contrast_score=0.7)
    hw = HardwareState(gpu_available=False)
    
    decision = agent.select_strategy(environment_state=env, hardware_state=hw, historical_knowledge=[])
    
    assert isinstance(decision, Decision)
    assert decision.selected_strategy in agent.strategies
    assert decision.constraints_satisfied is True
    assert decision.evidence_strength == 0.0
    assert "cold-start" in decision.explanation

def test_unknown_model_hardware_requirement_is_unevaluated(agent):
    env = EnvironmentState(haze_score=0.9, visibility_score=0.1)
    hw = HardwareState(gpu_available=False)
    
    decision = agent.select_strategy(environment_state=env, hardware_state=hw)
    
    assert decision.strategy_scores["DehazeFormer"] > 0.0
    assert "hardware_requirements" in decision.feasibility_info["DehazeFormer"]["unevaluated"]

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
    requirements = {"max_latency_ms": 100, "target_fps": 30}
    
    performance_data = {
        "DEA-Net": {"latency_ms": 150, "fps": 40},
        "DehazeFormer": {"latency_ms": 80, "fps": 20},
        "Bypass": {"latency_ms": 10, "fps": 60}
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
    requirements = {"max_latency_ms": 100, "target_fps": 30}
    
    decision = agent.select_strategy(
        environment_state=env, 
        hardware_state=hw, 
        requirements=requirements
    )
    
    assert "Criteria unevaluated due to missing data: hardware_requirements, latency, fps" in decision.explanation

def test_hardware_state_effect(agent):
    env = EnvironmentState(haze_score=0.5)
    hw_no_gpu = HardwareState(gpu_available=False)
    hw_with_gpu = HardwareState(gpu_available=True)
    
    decision_no_gpu = agent.select_strategy(environment_state=env, hardware_state=hw_no_gpu)
    decision_with_gpu = agent.select_strategy(environment_state=env, hardware_state=hw_with_gpu)
    
    assert decision_no_gpu.strategy_scores["DEA-Net"] < decision_with_gpu.strategy_scores["DEA-Net"]
    assert decision_no_gpu.selected_strategy in agent.strategies

def test_environment_fields_affect_scores_without_model_claims(agent):
    clear = EnvironmentState(haze_score=0.1, visibility_score=0.9, illumination_score=0.5, contrast_score=0.9)
    degraded = EnvironmentState(haze_score=0.9, visibility_score=0.1, illumination_score=0.1, contrast_score=0.1)
    assert agent._score_environment("Bypass", clear) > agent._score_environment("Bypass", degraded)
    assert agent._score_environment("DEA-Net", degraded) > agent._score_environment("DEA-Net", clear)
    assert agent._score_environment("DEA-Net", degraded) == agent._score_environment("DehazeFormer", degraded)

def test_low_medium_high_haze_and_visibility_affect_environment_score(agent):
    low = EnvironmentState(haze_score=0.1, visibility_score=0.8)
    medium = EnvironmentState(haze_score=0.5, visibility_score=0.5)
    high = EnvironmentState(haze_score=0.9, visibility_score=0.2)
    assert agent._score_environment("Bypass", low) > agent._score_environment("Bypass", medium)
    assert agent._score_environment("Bypass", medium) > agent._score_environment("Bypass", high)
    assert agent._score_environment("DEA-Net", high) > agent._score_environment("DEA-Net", low)
    assert agent._score_environment("Bypass", EnvironmentState(haze_score=0.5, visibility_score=0.9)) > agent._score_environment("Bypass", EnvironmentState(haze_score=0.5, visibility_score=0.1))

def test_partial_state_values_do_not_crash(agent):
    decision = agent.select_strategy(EnvironmentState(), HardwareState(gpu_available=False))
    assert decision.selected_strategy in agent.strategies

def test_hardware_metrics_are_consumed_safely(agent):
    busy = HardwareState(gpu_available=True, gpu_utilization=95, gpu_memory_available=128, cpu_utilization=95, ram_utilization=95)
    available = HardwareState(gpu_available=True, gpu_utilization=10, gpu_memory_available=8192, cpu_utilization=10, ram_utilization=10)
    assert agent._score_resource("DehazeFormer", available) > agent._score_resource("DehazeFormer", busy)

def test_synthetic_history_is_not_evidence(agent):
    decision = agent.select_strategy(historical_knowledge=[{"_comment": "These are synthetic test fixtures and are not experimental results.", "strategy": "DEA-Net", "outcome": {"success": True}}])
    assert decision.evidence_strength == 0.0
    assert "cold-start" in decision.explanation

def test_deterministic_output(agent):
    env = EnvironmentState(haze_score=0.5)
    hw = HardwareState(gpu_available=True)
    
    decision1 = agent.select_strategy(environment_state=env, hardware_state=hw)
    decision2 = agent.select_strategy(environment_state=env, hardware_state=hw)
    
    assert decision1.selected_strategy == decision2.selected_strategy
    assert decision1.strategy_scores == decision2.strategy_scores
