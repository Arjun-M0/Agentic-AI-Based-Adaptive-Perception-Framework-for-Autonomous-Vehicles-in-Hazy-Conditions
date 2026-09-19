from ultralytics import YOLO
from src.models.base_model import BaseModel


class YOLOv11Model(BaseModel):

    def __init__(self, model_path="weights/yolo11n.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = YOLO(self.model_path)

    def infer(self, image):
        if self.model is None:
            self.load_model()

        return self.model(image)