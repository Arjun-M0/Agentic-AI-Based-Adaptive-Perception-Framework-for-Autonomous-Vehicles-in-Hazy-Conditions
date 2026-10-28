"""Compare cold-start and knowledge-assisted routing on the same image.

The first pass makes a cold-start decision across all configured strategies,
executes it, and stores the measured outcome. The second pass retrieves that
experience and makes a knowledge-assisted decision for the same image.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.decision_agent import AdaptiveDecisionAgent
from src.agents.environment_agent import EnvironmentUnderstandingAgent
from src.agents.hardware_agent import HardwareAwarenessAgent
from src.agents.knowledge_agent import KnowledgeRepositoryAgent
from src.agents.restoration_agent import ImageRestorationAgent


def _median_latency(agent, image, strategy, repeats):
    agent.restore(image, strategy)
    samples = [agent.restore(image, strategy)[1] for _ in range(repeats)]
    return statistics.median(samples), samples


def _experience(environment, hardware, strategy, latency, image_path):
    return {
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
        "strategy": strategy,
        "performance": {"restoration_latency_ms": latency, "latency_ms": latency},
        "outcome": {"success": True},
        "provenance": {"image_path": str(image_path), "benchmark": True},
    }


def run_benchmark(image_path, knowledge_path, repeats):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    environment = EnvironmentUnderstandingAgent().analyze(image)
    hardware = HardwareAwarenessAgent().analyze()
    restoration = ImageRestorationAgent()

    knowledge = KnowledgeRepositoryAgent(str(knowledge_path))
    decision_agent = AdaptiveDecisionAgent()

    cold_decision_started = time.perf_counter()
    cold_decision = decision_agent.select_strategy(environment, hardware, [])
    cold_decision_latency = (time.perf_counter() - cold_decision_started) * 1000
    cold_latency, cold_samples = _median_latency(
        restoration, image, cold_decision.selected_strategy, repeats
    )
    if not knowledge.update(
        _experience(
            environment,
            hardware,
            cold_decision.selected_strategy,
            cold_latency,
            image_path,
        )
    ):
        raise OSError(f"Could not write benchmark experience to {knowledge_path}")

    retrieval_started = time.perf_counter()
    history = knowledge.retrieve(environment, hardware)
    retrieval_latency = (time.perf_counter() - retrieval_started) * 1000
    warm_decision_started = time.perf_counter()
    warm_decision = decision_agent.select_strategy(environment, hardware, history)
    warm_decision_latency = (time.perf_counter() - warm_decision_started) * 1000
    warm_latency, warm_samples = _median_latency(
        restoration, image, warm_decision.selected_strategy, repeats
    )

    cold_total = cold_decision_latency + cold_latency
    warm_total = retrieval_latency + warm_decision_latency + warm_latency
    reduction = (
        (cold_total - warm_total) / cold_total * 100
        if cold_total > 0
        else 0.0
    )
    return {
        "image": str(image_path),
        "repeats": repeats,
        "knowledge_repository": str(knowledge_path),
        "before_knowledge": {
            "definition": "cold-start adaptive decision with no historical experiences",
            "strategy": cold_decision.selected_strategy,
            "strategy_scores": cold_decision.strategy_scores,
            "history_count": 0,
            "decision_latency_ms": cold_decision_latency,
            "samples_ms": cold_samples,
            "median_restoration_latency_ms": cold_latency,
            "total_latency_ms": cold_total,
            "decision_explanation": cold_decision.explanation,
        },
        "after_knowledge": {
            "definition": "knowledge retrieval + historical-guided scoring on the same image",
            "strategy": warm_decision.selected_strategy,
            "strategy_scores": warm_decision.strategy_scores,
            "history_count": len(history),
            "retrieval_latency_ms": retrieval_latency,
            "decision_latency_ms": warm_decision_latency,
            "samples_ms": warm_samples,
            "median_restoration_latency_ms": warm_latency,
            "total_latency_ms": warm_total,
            "decision_explanation": warm_decision.explanation,
        },
        "latency_reduction_percent": reduction,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark knowledge-assisted latency.")
    parser.add_argument("--image", default="data/samples/test_road.jpg")
    parser.add_argument("--knowledge", default="knowledge/restoration_history.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="outputs/reports/knowledge_latency_comparison.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    result = run_benchmark(
        Path(args.image), Path(args.knowledge), args.repeats
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
