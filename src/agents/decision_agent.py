from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import yaml
import logging
from src.agents.base_agent import BaseAgent
from src.core.context import EnvironmentState, HardwareState

logger = logging.getLogger(__name__)

@dataclass
class Decision:
    selected_strategy: str
    strategy_scores: Dict[str, float]
    explanation: str
    constraints_satisfied: bool
    evidence_strength: float

class AdaptiveDecisionAgent(BaseAgent):
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.strategies = self.config.get("decision", {}).get("strategies", ["Bypass", "DEA-Net", "DehazeFormer"])
        self.weights = self.config.get("decision", {}).get("weights", {
            "environment": 0.4,
            "resource": 0.3,
            "historical": 0.3,
            "performance": 0.0
        })

    def _load_config(self) -> dict:
        import os
        if not os.path.exists(self.config_path):
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _evaluate_feasibility(self, strategy: str, hardware_state: Optional[HardwareState], requirements: Dict[str, Any]) -> bool:
        if strategy == "Bypass":
            return True
            
        if not hardware_state:
            return True
            
        if strategy == "DehazeFormer":
            if hardware_state.gpu_available is False:
                return False
                
        return True

    def _score_environment(self, strategy: str, environment_state: Optional[EnvironmentState]) -> float:
        if not environment_state or environment_state.haze_score is None:
            return 0.5
            
        haze = environment_state.haze_score
        if strategy == "Bypass":
            return max(0.0, 1.0 - (haze * 2))
        elif strategy == "DEA-Net":
            return 1.0 - abs(haze - 0.5)
        elif strategy == "DehazeFormer":
            return haze
        return 0.5

    def _score_resource(self, strategy: str, hardware_state: Optional[HardwareState]) -> float:
        if not hardware_state:
            return 0.5
            
        has_gpu = hardware_state.gpu_available
        if strategy == "Bypass":
            return 1.0
        elif strategy == "DEA-Net":
            return 0.8 if has_gpu else 0.4
        elif strategy == "DehazeFormer":
            return 0.6 if has_gpu else 0.1
        return 0.5

    def _score_historical(self, strategy: str, historical_knowledge: List[Dict[str, Any]]) -> float:
        if not historical_knowledge:
            return 0.5
            
        strategy_exps = [exp for exp in historical_knowledge if exp.get("strategy") == strategy]
        if not strategy_exps:
            return 0.5
            
        success_count = sum(1 for exp in strategy_exps if exp.get("outcome", {}).get("success", False))
        return success_count / len(strategy_exps)

    def select_strategy(self, 
                        environment_state: Optional[EnvironmentState] = None, 
                        hardware_state: Optional[HardwareState] = None, 
                        historical_knowledge: Optional[List[Dict[str, Any]]] = None,
                        requirements: Optional[Dict[str, Any]] = None) -> Decision:
        """
        Selects the best restoration strategy based on state, constraints, and historical knowledge.
        """
        historical_knowledge = historical_knowledge or []
        requirements = requirements or self.config.get("decision", {}).get("requirements", {})
        
        scores = {}
        feasible_strategies = []
        infeasible_strategies = []
        
        for strategy in self.strategies:
            feasible = self._evaluate_feasibility(strategy, hardware_state, requirements)
            if not feasible:
                scores[strategy] = 0.0
                infeasible_strategies.append(strategy)
                continue
                
            feasible_strategies.append(strategy)
            
            env_score = self._score_environment(strategy, environment_state)
            res_score = self._score_resource(strategy, hardware_state)
            hist_score = self._score_historical(strategy, historical_knowledge)
            
            total_score = (
                env_score * self.weights.get("environment", 0.0) +
                res_score * self.weights.get("resource", 0.0) +
                hist_score * self.weights.get("historical", 0.0)
            )
            scores[strategy] = total_score
            
        if not feasible_strategies:
            selected_strategy = "Bypass"
            constraints_satisfied = False
            explanation = "No feasible strategy found. Defaulting to Bypass."
        else:
            selected_strategy = max(feasible_strategies, key=lambda s: scores[s])
            constraints_satisfied = True
            
            if not historical_knowledge:
                explanation = f"No historical experiences were available; selection was based on current environment, resource feasibility, configured strategy priors, and available performance evidence. Selected {selected_strategy}."
            else:
                reasons = []
                if infeasible_strategies:
                    reasons.append(f"{', '.join(infeasible_strategies)} marked infeasible due to hardware/latency constraints")
                reasons.append(f"{selected_strategy} achieved highest score based on environment, resources, and history")
                explanation = f"{selected_strategy} selected. " + "; ".join(reasons) + "."

        evidence_strength = min(1.0, len(historical_knowledge) / 10.0)

        decision = Decision(
            selected_strategy=selected_strategy,
            strategy_scores=scores,
            explanation=explanation,
            constraints_satisfied=constraints_satisfied,
            evidence_strength=evidence_strength
        )
        
        logger.info(f"Decision made: {decision.selected_strategy} - {decision.explanation}")
        return decision
