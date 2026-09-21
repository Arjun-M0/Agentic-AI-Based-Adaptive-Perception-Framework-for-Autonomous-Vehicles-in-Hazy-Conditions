import numpy as np
import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from .backbone import Backbone

def pad_img(x, patch_size=4):
    _, _, h, w = x.size()

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    return F.pad(
        x,
        (0, pad_w, 0, pad_h),
        mode="reflect"
    )

class DEANetModel(BaseModel):

    def __init__(self, weight_path):
        self.weight_path = weight_path

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None

    def load_model(self):
        self.model = Backbone()

        checkpoint = torch.load(
            self.weight_path,
            map_location="cpu"
        )
        
        state_dict_raw = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        self.model.load_state_dict(state_dict_raw)

        self.model = self.model.to(self.device)

        self.model.eval()

    def infer(self, image):
        if self.model is None:
            raise RuntimeError("DEA-Net model is not loaded.")

    # BGR → RGB
        image_rgb = image[:, :, ::-1].copy()

    # 0–255 integers → 0–1 float values
        image_float = image_rgb.astype(np.float32) / 255.0

    # NumPy → PyTorch tensor
        image_tensor = torch.from_numpy(image_float)

    # HWC → CHW
        image_tensor = image_tensor.permute(2, 0, 1)

    # CHW → BCHW
        image_tensor = image_tensor.unsqueeze(0)

    # Move image to CPU/GPU
        image_tensor = image_tensor.to(self.device)

    # Remember original image size
        height, width = image_tensor.shape[2:]

    # Make height and width compatible with DEA-Net
        padded_tensor = pad_img(image_tensor, 4)

    # Run DEA-Net without training calculations
        with torch.no_grad():
            output = self.model(padded_tensor)

    # Keep valid image values only
        output = output.clamp(0, 1)

    # Remove the padding we added earlier
        output = output[:, :, :height, :width]

        output = output.squeeze(0)

        output = output.permute(1, 2, 0)

        output = output.cpu().numpy()

        output = (output * 255.0).clip(0, 255).astype(np.uint8)

        output = output[:, :, ::-1].copy()

        return output