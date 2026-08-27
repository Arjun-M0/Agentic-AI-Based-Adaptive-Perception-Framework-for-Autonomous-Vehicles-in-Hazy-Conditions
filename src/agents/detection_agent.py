from src.agents.base_agent import BaseAgent

class ObjectDetectionAgent(BaseAgent):
    def detect(self, image):
        """
        Runs object detection (e.g., YOLOv11) on the provided image.
        """
        raise NotImplementedError('Object detection is not yet implemented.')
