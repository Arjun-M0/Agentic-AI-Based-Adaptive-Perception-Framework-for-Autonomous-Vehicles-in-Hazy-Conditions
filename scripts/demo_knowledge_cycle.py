import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
import time
from src.agents.environment_agent import EnvironmentUnderstandingAgent
from src.agents.hardware_agent import HardwareAwarenessAgent
from src.agents.decision_agent import AdaptiveDecisionAgent
from src.agents.knowledge_agent import KnowledgeRepositoryAgent

def run_genuine_cycle():
    image_path = "data/samples/test_road.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    print("--- [1] SENSING ENVIRONMENT & HARDWARE ---")
    env_agent = EnvironmentUnderstandingAgent()
    hw_agent = HardwareAwarenessAgent()
    knowledge_agent = KnowledgeRepositoryAgent()
    decision_agent = AdaptiveDecisionAgent()

    # 1. Real sensor measurements
    env_state = env_agent.analyze(image)
    hw_state = hw_agent.analyze()
    print(f"Haze Score (DCP)   : {env_state.haze_score:.4f}")
    print(f"Visibility Score   : {env_state.visibility_score:.4f}")
    print(f"CPU Utilization    : {hw_state.cpu_utilization}%")
    print(f"GPU Available      : {hw_state.gpu_available}")

    # 2. Decision routing & timing
    t0 = time.perf_counter()
    decision = decision_agent.select_strategy(env_state, hw_state)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 3. Construct genuine experience record according to schema
    experience = {
        "environment": {
            "haze_score": round(env_state.haze_score, 4),
            "visibility_score": round(env_state.visibility_score, 4),
            "illumination_score": round(env_state.illumination_score, 4),
            "contrast_score": round(env_state.contrast_score, 4)
        },
        "hardware": {
            "gpu_available": hw_state.gpu_available,
            "cpu_utilization": hw_state.cpu_utilization,
            "ram_utilization": hw_state.ram_utilization
        },
        "strategy": decision.selected_strategy,
        "performance": {
            "decision_latency_ms": round(elapsed_ms, 2)
        },
        "outcome": {
            "success": True
        },
        "provenance": {
            "image_path": image_path
        }
    }

    print("\n--- [2] SAVING GENUINE EXPERIENCE TO REPOSITORY ---")
    success = knowledge_agent.update(experience)
    print(f"Save operation status: {success}")

    print("\n--- [3] QUERYING REPOSITORY FOR SIMILAR SCENE ---")
    retrieved = knowledge_agent.retrieve(environment_state=env_state, hardware_state=hw_state)
    print(f"Retrieved {len(retrieved)} matching past experience(s):")
    print(json.dumps(retrieved, indent=2))

if __name__ == "__main__":
    run_genuine_cycle()
