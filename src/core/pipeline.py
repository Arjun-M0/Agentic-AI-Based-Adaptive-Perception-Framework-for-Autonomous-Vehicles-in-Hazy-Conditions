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
            decision = self.decision_agent.select_strategy(
                environment_state=environment,
                hardware_state=hardware,
                historical_knowledge=history,
            )
            restored, restoration_latency = self.restoration_agent.restore(
                image, decision.selected_strategy
            )

            annotated_path = None
            if output_path:
                annotated_path = str(output_path / f"{Path(item['filename']).stem}_detections.jpg")
            detection = self.detection_agent.detect(restored, annotated_path)
            evaluation = self.evaluation_agent.evaluate(
                {
                    "latency_ms": restoration_latency + detection["latency_ms"],
                    "fps": 1000 / (restoration_latency + detection["latency_ms"])
                    if restoration_latency + detection["latency_ms"] > 0
                    else 0,
                }
            )

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
                }
            )

        return {"source": str(source), "results": results, "skipped": skipped}
