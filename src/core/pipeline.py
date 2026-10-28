import logging
from pathlib import Path
from typing import Dict, Optional

from src.agents.decision_agent import AdaptiveDecisionAgent
from src.agents.detection_agent import ObjectDetectionAgent
from src.agents.environment_agent import EnvironmentUnderstandingAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.hardware_agent import HardwareAwarenessAgent
from src.agents.knowledge_agent import KnowledgeRepositoryAgent
from src.agents.restoration_agent import ImageRestorationAgent
from src.input.input_module import InputModule

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """Runs input, analysis, adaptive processing, detection, and evaluation."""

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        knowledge_path: str = "knowledge/restoration_history.json",
        detection_model_path: str = "weights/yolo11n.pt",
    ):
        self.input_module = InputModule()
        self.environment_agent = EnvironmentUnderstandingAgent()
        self.hardware_agent = HardwareAwarenessAgent()
        self.decision_agent = AdaptiveDecisionAgent(config_path)
        self.knowledge_agent = KnowledgeRepositoryAgent(knowledge_path)
        self.restoration_agent = ImageRestorationAgent()
        self.detection_agent = ObjectDetectionAgent(detection_model_path)
        self.evaluation_agent = EvaluationAgent()

    def run(self, source: str, output_dir: Optional[str] = None) -> Dict:
        """Run the pipeline for one image or all supported images under a directory."""
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        results = []
        skipped = []
        for frame_id, item in enumerate(self.input_module.process(source), start=1):
            if item["image"] is None:
                skipped.append(
                    {"filename": item["filename"], "path": item["path"], "error": item["error"]}
                )
                logger.warning("Skipping %s: %s", item["filename"], item["error"])
                continue

            image = item["image"]
            environment = self.environment_agent.analyze(image)
            hardware = self.hardware_agent.analyze()
            history = self.knowledge_agent.retrieve(environment, hardware)
            similar_history = self.knowledge_agent.has_similar_experience(
                environment, hardware
            )
            candidate_runs = {}
            strategies = (
                self.decision_agent.strategies if not similar_history
                else [None]
            )
            for strategy in strategies:
                if strategy is None:
                    decision = self.decision_agent.select_strategy(
                        environment_state=environment,
                        hardware_state=hardware,
                        historical_knowledge=history,
                    )
                    strategy = decision.selected_strategy
                restored, restoration_latency = self.restoration_agent.restore(
                    image, strategy
                )
                annotated_path = None
                if output_path:
                    suffix = f"_{strategy.lower().replace('-', '_')}" if not similar_history else ""
                    annotated_path = str(
                        output_path
                        / f"{Path(item['filename']).stem}{suffix}_detections.jpg"
                    )
                detection = self.detection_agent.detect(restored, annotated_path)
                total_latency = restoration_latency + detection["latency_ms"]
                candidate_runs[strategy] = {
                    "restored": restored,
                    "restoration_latency": restoration_latency,
                    "detection": detection,
                    "evaluation": self.evaluation_agent.evaluate(
                        {
                            "latency_ms": total_latency,
                            "fps": 1000 / total_latency if total_latency > 0 else 0,
                        }
                    ),
                }

            if not similar_history:
                performance_data = {
                    strategy: {
                        "latency_ms": candidate["evaluation"]["latency_ms"],
                        "fps": candidate["evaluation"]["fps"],
                    }
                    for strategy, candidate in candidate_runs.items()
                }
                decision = self.decision_agent.select_strategy(
                    environment_state=environment,
                    hardware_state=hardware,
                    historical_knowledge=history,
                    performance_data=performance_data,
                )
            candidate = candidate_runs[decision.selected_strategy]
            candidate_latencies = {
                strategy: candidate["evaluation"]["latency_ms"]
                for strategy, candidate in candidate_runs.items()
            }
            exploration_total_latency = (
                sum(candidate_latencies.values()) if not similar_history else None
            )
            previous_exploration_latency = next(
                (
                    experience.get("performance", {}).get(
                        "cold_start_exploration_total_latency_ms"
                    )
                    for experience in history
                    if isinstance(
                        experience.get("performance", {}).get(
                            "cold_start_exploration_total_latency_ms"
                        ),
                        (int, float),
                    )
                ),
                None,
            )
            previous_candidate_latency_sets = [
                experience.get("performance", {}).get("candidate_latencies_ms")
                for experience in history
                if isinstance(
                    experience.get("performance", {}).get(
                        "candidate_latencies_ms"
                    ),
                    dict,
                )
            ]
            previous_candidate_latencies = (
                max(previous_candidate_latency_sets, key=len)
                if previous_candidate_latency_sets
                else None
            )
            baseline_exploration_latency = (
                exploration_total_latency
                if exploration_total_latency is not None
                else previous_exploration_latency
            )
            current_total_latency = candidate["evaluation"]["latency_ms"]
            latency_reduction = (
                (
                    baseline_exploration_latency - current_total_latency
                )
                / baseline_exploration_latency
                * 100
                if (
                    similar_history
                    and baseline_exploration_latency
                    and baseline_exploration_latency > 0
                )
                else None
            )
            restored = candidate["restored"]
            restoration_latency = candidate["restoration_latency"]
            detection = candidate["detection"]
            evaluation = candidate["evaluation"]
            knowledge_update = self.knowledge_agent.update(
                {
                    "environment": {
                        "haze_score": environment.haze_score,
                        "visibility_score": environment.visibility_score,
                        "illumination_score": environment.illumination_score,
                        "contrast_score": environment.contrast_score,
                    },
                    "hardware": {
                        "gpu_available": hardware.gpu_available,
                        "gpu_utilization": hardware.gpu_utilization,
                        "gpu_memory_available": hardware.gpu_memory_available,
                        "cpu_utilization": hardware.cpu_utilization,
                        "ram_utilization": hardware.ram_utilization,
                    },
                    "strategy": decision.selected_strategy,
                    "performance": {
                        "restoration_latency_ms": restoration_latency,
                        "detection_latency_ms": detection["latency_ms"],
                        "latency_ms": evaluation["latency_ms"],
                        "fps": evaluation["fps"],
                        "candidate_latencies_ms": candidate_latencies,
                        "previous_candidate_latencies_ms": previous_candidate_latencies,
                        "cold_start_exploration_total_latency_ms": baseline_exploration_latency,
                        "knowledge_assisted_latency_ms": (
                            current_total_latency if similar_history else None
                        ),
                    },
                    "outcome": {
                        "success": True,
                        "detection_count": len(detection["detections"]),
                    },
                    "provenance": {
                        "image_path": item["path"],
                        "cold_start_exploration": not similar_history,
                        "candidate_strategies": list(candidate_runs),
                    },
                }
            )
            if not knowledge_update:
                logger.error("Could not persist knowledge for %s", item["filename"])

            results.append(
                {
                    "frame_id": frame_id,
                    "filename": item["filename"],
                    "path": item["path"],
                    "width": item["width"],
                    "height": item["height"],
                    "environment": environment,
                    "hardware": hardware,
                    "decision": decision,
                    "restoration_latency_ms": restoration_latency,
                    "detection": detection,
                    "evaluation": evaluation,
                    "knowledge_count": len(history),
                    "cold_start_exploration": not similar_history,
                    "candidate_count": len(candidate_runs),
                    "candidate_latencies_ms": candidate_latencies,
                    "cold_start_exploration_total_latency_ms": baseline_exploration_latency,
                    "previous_exploration_latency_ms": previous_exploration_latency,
                    "knowledge_assisted_latency_ms": (
                        current_total_latency if similar_history else None
                    ),
                    "latency_reduction_percent": latency_reduction,
                    "knowledge_update": knowledge_update,
                }
            )

        return {"source": str(source), "results": results, "skipped": skipped}
