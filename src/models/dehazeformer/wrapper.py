import numpy as np
import torch

from src.models.base_model import BaseModel
from .dehazeformer import dehazeformer_t


class DehazeFormerModel(BaseModel):

    def __init__(self, weight_path):
        self.weight_path = weight_path

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None

    def load_model(self):
        self.model = dehazeformer_t()

        checkpoint = torch.load(
            self.weight_path,
            map_location="cpu"
        )

        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint["state_dict"].items()
        }

        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)

        self.model.eval()

    def infer(self, image):
        if self.model is None:
            raise RuntimeError(
                "DehazeFormer model is not loaded."
            )

        # BGR -> RGB
        image_rgb = image[:, :, ::-1].copy()

        # uint8 0-255 -> float32 0-1
        image_float = image_rgb.astype(np.float32) / 255.0

        # NumPy -> PyTorch
        image_tensor = torch.from_numpy(image_float)

        # HWC -> CHW
        image_tensor = image_tensor.permute(2, 0, 1)

        # CHW -> BCHW
        image_tensor = image_tensor.unsqueeze(0)

        # Move image to CPU/GPU
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)

        output = output.clamp(0, 1)

        # BCHW -> CHW
        output = output.squeeze(0)

        # CHW -> HWC
        output = output.permute(1, 2, 0)

        # GPU tensor -> NumPy
        output = output.cpu().numpy()

        # 0-1 -> 0-255
        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        # RGB -> BGR
        output = output[:, :, ::-1].copy()

        return output