import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.decision_agent import AdaptiveDecisionAgent
from src.agents.detection_agent import ObjectDetectionAgent
from src.agents.environment_agent import EnvironmentUnderstandingAgent
from src.agents.hardware_agent import HardwareAwarenessAgent
from src.agents.knowledge_agent import KnowledgeRepositoryAgent
from src.agents.restoration_agent import ImageRestorationAgent


def print_stage(name, value):
	print(f"\n=== {name} OUTPUT ===", file=sys.stderr)
	print(json.dumps(value, indent=2), file=sys.stderr)


def main():
	parser = argparse.ArgumentParser(description="Run the perception agents on one image.")
	parser.add_argument("image", help="Path to a road image")
	parser.add_argument(
		"--strategy",
		choices=("auto", "bypass", "deanet", "dehazeformer"),
		default="auto",
		help="Restoration strategy; auto uses the decision agent",
	)
	parser.add_argument(
		"--output",
		default="outputs/images/restored_road.jpg",
		help="Path for the restored image",
	)
	parser.add_argument(
		"--yolo-weights",
		default="weights/yolo11n.pt",
		help="Path to YOLO weights",
	)
	parser.add_argument(
		"--detection-output",
		default="outputs/images/detections_road.jpg",
		help="Path for the annotated YOLO detection image",
	)
	args = parser.parse_args()

	image = cv2.imread(args.image)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {args.image}")

	print("[1/6] Running Environment Understanding Agent...", file=sys.stderr)
	environment = EnvironmentUnderstandingAgent().analyze(image)
	print_stage("ENVIRONMENT AGENT", asdict(environment))
	print("[2/6] Running Hardware Awareness Agent...", file=sys.stderr)
	hardware = HardwareAwarenessAgent().analyze()
	print_stage("HARDWARE AGENT", asdict(hardware))
	print("[3/6] Running Knowledge Repository Agent...", file=sys.stderr)
	knowledge = KnowledgeRepositoryAgent().retrieve(environment, hardware)
	print_stage("KNOWLEDGE AGENT", knowledge)
	print("[4/6] Running Adaptive Decision Agent...", file=sys.stderr)
	decision = AdaptiveDecisionAgent().select_strategy(environment, hardware, knowledge)
	print_stage("DECISION AGENT", asdict(decision))

	strategy = decision.selected_strategy.lower() if args.strategy == "auto" else args.strategy
	print(f"[5/6] Running Restoration Agent ({strategy})...", file=sys.stderr)
	restoration_started = time.perf_counter()
	restored_image, restoration_latency = ImageRestorationAgent().restore(image, strategy)
	restoration_wall_time = (time.perf_counter() - restoration_started) * 1000
	print_stage("RESTORATION AGENT", {
		"strategy": strategy,
		"shape": list(restored_image.shape),
		"dtype": str(restored_image.dtype),
		"latency_ms": restoration_latency,
	})

	os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
	if not cv2.imwrite(args.output, restored_image):
		raise OSError(f"Could not write restored image: {args.output}")

	print("[6/6] Running Object Detection Agent...", file=sys.stderr)
	os.makedirs(os.path.dirname(args.detection_output) or ".", exist_ok=True)
	detection = ObjectDetectionAgent(args.yolo_weights).detect(
		restored_image,
		annotated_output_path=args.detection_output,
	)
	print_stage("DETECTION AGENT", detection)

	output = {
		"input": {"path": args.image, "shape": list(image.shape)},
		"environment": asdict(environment),
		"hardware": asdict(hardware),
		"knowledge_count": len(knowledge),
		"decision": asdict(decision),
		"restoration": {
			"strategy": strategy,
			"shape": list(restored_image.shape),
			"latency_ms": restoration_latency,
			"wall_time_ms": restoration_wall_time,
			"output_path": args.output,
		},
		"detection": detection,
		"detection_output_path": args.detection_output,
	}
	print(json.dumps(output, indent=2))


if __name__ == "__main__":
	main()
