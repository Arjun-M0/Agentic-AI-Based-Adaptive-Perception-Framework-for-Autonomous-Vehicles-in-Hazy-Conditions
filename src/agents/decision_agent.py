from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
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
    feasibility_info: Dict[str, Any] = field(default_factory=dict)

class AdaptiveDecisionAgent(BaseAgent):
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.strategies = self.config.get("decision", {}).get("strategies", ["Bypass", "DEA-Net", "DehazeFormer"])
        self.weights = self.config.get("decision", {}).get("weights", {
            "environment": 0.2,
            "resource": 0.2,
            "historical": 0.6,
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

    def _evaluate_feasibility(self, strategy: str, hardware_state: Optional[HardwareState], requirements: Dict[str, Any], performance_data: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[bool, str, List[str], List[str]]:
        evaluated = []
        unevaluated = []
        
        if strategy == "DehazeFormer":
            evaluated.append("hardware")
            if hardware_state and hardware_state.gpu_available is False:
                return False, "Requires GPU, but GPU is unavailable", evaluated, unevaluated
        else:
            evaluated.append("hardware")
            
        if "max_latency" in requirements:
            if performance_data and strategy in performance_data and "latency" in performance_data[strategy]:
                evaluated.append("latency")
                if performance_data[strategy]["latency"] > requirements["max_latency"]:
                    return False, f"Latency {performance_data[strategy]['latency']} > max {requirements['max_latency']}", evaluated, unevaluated
            else:
                unevaluated.append("latency")
                
        if "target_fps" in requirements:
            if performance_data and strategy in performance_data and "fps" in performance_data[strategy]:
                evaluated.append("fps")
                if performance_data[strategy]["fps"] < requirements["target_fps"]:
                    return False, f"FPS {performance_data[strategy]['fps']} < target {requirements['target_fps']}", evaluated, unevaluated
            else:
                unevaluated.append("fps")
                
        return True, "Feasible", evaluated, unevaluated

    def _score_environment(self, strategy: str, environment_state: Optional[EnvironmentState]) -> float:
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

    def _score_historical(self, strategy: str, historical_knowledge: List[Dict[str, Any]], current_env: Optional[EnvironmentState] = None) -> float:
        if not historical_knowledge:
            return 0.5
            
        strategy_exps = [exp for exp in historical_knowledge if exp.get("strategy") == strategy]
        if not strategy_exps:
            return 0.5
            
        relevant_exps = strategy_exps
        if current_env and current_env.haze_score is not None:
            similar_exps = []
            for exp in strategy_exps:
                env_data = exp.get("environment", {})
                if isinstance(env_data, dict) and "haze_score" in env_data:
                    if abs(env_data["haze_score"] - current_env.haze_score) <= 0.2:
                        similar_exps.append(exp)
            if similar_exps:
                relevant_exps = similar_exps
                
        success_count = sum(1 for exp in relevant_exps if exp.get("outcome", {}).get("success", False))
        return success_count / len(relevant_exps)

    def select_strategy(self, 
                        environment_state: Optional[EnvironmentState] = None, 
                        hardware_state: Optional[HardwareState] = None, 
                        historical_knowledge: Optional[List[Dict[str, Any]]] = None,
                        requirements: Optional[Dict[str, Any]] = None,
                        performance_data: Optional[Dict[str, Dict[str, float]]] = None) -> Decision:
        historical_knowledge = historical_knowledge or []
        requirements = requirements or self.config.get("decision", {}).get("requirements", {})
        
        scores = {}
        feasible_strategies = []
        infeasible_strategies = {}
        strategy_feasibility_info = {}
        
        for strategy in self.strategies:
            is_feasible, reason, eval_const, uneval_const = self._evaluate_feasibility(
                strategy, hardware_state, requirements, performance_data
            )
            strategy_feasibility_info[strategy] = {
                "feasible": is_feasible,
                "reason": reason,
                "evaluated": eval_const,
                "unevaluated": uneval_const
            }
            
            if not is_feasible:
                scores[strategy] = 0.0
                infeasible_strategies[strategy] = reason
                continue
                
            feasible_strategies.append(strategy)
            
            env_score = self._score_environment(strategy, environment_state)
            res_score = self._score_resource(strategy, hardware_state)
            hist_score = self._score_historical(strategy, historical_knowledge, environment_state)
            
            total_score = (
                env_score * self.weights.get("environment", 0.0) +
                res_score * self.weights.get("resource", 0.0) +
                hist_score * self.weights.get("historical", 0.0)
            )
            weight_sum = sum(self.weights.values())
            if weight_sum > 0:
                total_score /= weight_sum
                
            scores[strategy] = total_score
            
        if not feasible_strategies:
            selected_strategy = "Bypass"
            constraints_satisfied = False
            explanation = "No feasible strategy found under current constraints. Defaulting to Bypass as failsafe."
        else:
            selected_strategy = max(feasible_strategies, key=lambda s: scores[s])
            constraints_satisfied = True
            
            info = strategy_feasibility_info[selected_strategy]
            
            reasons = []
            if infeasible_strategies:
                reasons.append(f"Infeasible strategies: {', '.join([f'{s} ({r})' for s, r in infeasible_strategies.items()])}")
                
            if info["unevaluated"]:
                reasons.append(f"Constraints unverified due to missing data: {', '.join(info['unevaluated'])}")
            if info["evaluated"]:
                reasons.append(f"Satisfied evaluated constraints: {', '.join(info['evaluated'])}")
                
            if not historical_knowledge:
                reasons.append("No historical evidence was available (cold-start)")
            else:
                reasons.append("Historical evidence applied")
                
            explanation = f"Selected {selected_strategy}. " + "; ".join(reasons) + "."

        evidence_strength = min(1.0, len(historical_knowledge) / 10.0)

        decision = Decision(
            selected_strategy=selected_strategy,
            strategy_scores=scores,
            explanation=explanation,
            constraints_satisfied=constraints_satisfied,
            evidence_strength=evidence_strength,
            feasibility_info=strategy_feasibility_info
        )
        
        logger.info(f"Decision made: {decision.selected_strategy} - {decision.explanation}")
        return decision
