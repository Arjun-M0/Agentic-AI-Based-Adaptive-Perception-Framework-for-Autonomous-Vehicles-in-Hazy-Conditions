from src.agents.base_agent import BaseAgent
from src.models.yolov11.yolov11_model import YOLOv11Model
import cv2
import time


class ObjectDetectionAgent(BaseAgent):

    def __init__(self, model_path="weights/yolo11n.pt"):
        self.model = YOLOv11Model(model_path)

    def detect(self, image, annotated_output_path=None):
        start_time = time.perf_counter()

        results = self.model.infer(image)

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(box.conf[0]),
                    "box": box.xyxy[0].tolist()
                })

        if annotated_output_path:
            annotated_image = results[0].plot() if results else image
            if not cv2.imwrite(annotated_output_path, annotated_image):
                raise OSError(f"Could not write detection image: {annotated_output_path}")

        latency_ms = (time.perf_counter() - start_time) * 1000

        fps = 1000 / latency_ms if latency_ms > 0 else 0

        return {
            "detections": detections,
            "latency_ms": latency_ms,
            "fps": fps
        }