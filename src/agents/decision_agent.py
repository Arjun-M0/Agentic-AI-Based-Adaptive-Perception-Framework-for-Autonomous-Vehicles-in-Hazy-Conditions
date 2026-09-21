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

        # No strategy-specific hardware requirements have been measured yet.
        unevaluated.append("hardware_requirements")

        if "max_latency_ms" in requirements:
            if performance_data and strategy in performance_data and "latency_ms" in performance_data[strategy]:
                evaluated.append("latency")
                if performance_data[strategy]["latency_ms"] > requirements["max_latency_ms"]:
                    return False, f"Latency {performance_data[strategy]['latency_ms']} > max {requirements['max_latency_ms']}", evaluated, unevaluated
            else:
                unevaluated.append("latency")

        if "target_fps" in requirements:
            if performance_data and strategy in performance_data and "fps" in performance_data[strategy]:
                evaluated.append("fps")
                if performance_data[strategy]["fps"] < requirements["target_fps"]:
                    return False, f"FPS {performance_data[strategy]['fps']} < target {requirements['target_fps']}", evaluated, unevaluated
            else:
                unevaluated.append("fps")
                
        reason = "Verified constraints satisfied" if not unevaluated else "Verified constraints satisfied; some criteria unevaluated"
        return True, reason, evaluated, unevaluated

    def _score_environment(self, strategy: str, environment_state: Optional[EnvironmentState]) -> float:
        if environment_state is None:
            return 0.5

        values = {
            "haze": self._bounded(environment_state.haze_score),
            "visibility": self._bounded(environment_state.visibility_score),
            "illumination": self._bounded(environment_state.illumination_score),
            "contrast": self._bounded(environment_state.contrast_score),
        }
        if not any(value is not None for value in values.values()):
            return 0.5

        quality_values = [
            value for value in (
                1.0 - values["haze"] if values["haze"] is not None else None,
                values["visibility"],
                values["contrast"],
            ) if value is not None
        ]
        if values["illumination"] is not None:
            # Both very dark and saturated-bright frames are less useful for perception.
            quality_values.append(1.0 - abs(values["illumination"] - 0.5) * 2.0)

        environment_quality = sum(quality_values) / len(quality_values)
        if strategy == "Bypass":
            return self._bounded(environment_quality)
        if strategy in {"DEA-Net", "DehazeFormer"}:
            # This describes degraded-environment suitability, not measured model quality.
            return self._bounded(1.0 - environment_quality)
        return 0.5

    def _score_resource(self, strategy: str, hardware_state: Optional[HardwareState]) -> float:
        if not hardware_state:
            return 0.5

        if strategy == "Bypass":
            return 1.0

        signals = [1.0 if hardware_state.gpu_available else 0.0]
        if hardware_state.gpu_available and hardware_state.gpu_utilization is not None:
            signals.append(1.0 - self._bounded(hardware_state.gpu_utilization / 100.0))
        if hardware_state.gpu_available and hardware_state.gpu_memory_available is not None:
            signals.append(self._bounded(hardware_state.gpu_memory_available / 4096.0))
        for utilization in (hardware_state.cpu_utilization, hardware_state.ram_utilization):
            if utilization is not None:
                signals.append(1.0 - self._bounded(utilization / 100.0))

        availability = sum(signals) / len(signals)
        if strategy == "DehazeFormer":
            return self._bounded(availability * 0.9)
        if strategy == "DEA-Net":
            return self._bounded(availability * 0.95)
        return 0.5

    @staticmethod
    def _bounded(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))

    def _score_historical(self, strategy: str, historical_knowledge: List[Dict[str, Any]], current_env: Optional[EnvironmentState] = None) -> float:
        usable_history = [exp for exp in historical_knowledge if not exp.get("_comment", "").lower().startswith("these are synthetic")]
        if not usable_history:
            return 0.5
            
        strategy_exps = [exp for exp in usable_history if exp.get("strategy") == strategy]
        if not strategy_exps:
            return 0.5
            
        relevant_exps = strategy_exps
        if current_env:
            similar_exps = []
            current_values = {
                "haze_score": current_env.haze_score,
                "visibility_score": current_env.visibility_score,
                "illumination_score": current_env.illumination_score,
                "contrast_score": current_env.contrast_score,
            }
            for exp in strategy_exps:
                env_data = exp.get("environment", {})
                if not isinstance(env_data, dict):
                    continue
                compatible = [
                    abs(env_data[field] - value) <= 0.2
                    for field, value in current_values.items()
                    if value is not None and isinstance(env_data.get(field), (int, float))
                ]
                if compatible and all(compatible):
                    similar_exps.append(exp)
            if similar_exps:
                relevant_exps = similar_exps
                
        outcomes = [exp.get("outcome", {}).get("success") for exp in relevant_exps]
        measured_outcomes = [outcome for outcome in outcomes if isinstance(outcome, bool)]
        if not measured_outcomes:
            return 0.5
        return sum(measured_outcomes) / len(measured_outcomes)

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
                reasons.append(f"Criteria unevaluated due to missing data: {', '.join(info['unevaluated'])}")
            if info["evaluated"]:
                reasons.append(f"Satisfied evaluated constraints: {', '.join(info['evaluated'])}")
                
            usable_history = [exp for exp in historical_knowledge if not exp.get("_comment", "").lower().startswith("these are synthetic")]
            if not usable_history:
                reasons.append("No usable historical evidence was available (cold-start)")
            else:
                reasons.append("Historical evidence applied")
                
            explanation = f"Selected {selected_strategy} because it received the highest combined score from environment suitability, hardware/resource suitability, and historical evidence. " + "; ".join(reasons) + "."

        usable_history = [exp for exp in historical_knowledge if not exp.get("_comment", "").lower().startswith("these are synthetic")]
        evidence_strength = min(1.0, len(usable_history) / 10.0)

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
