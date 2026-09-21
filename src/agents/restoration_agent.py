import time
import torch
import numpy as np
from src.agents.base_agent import BaseAgent
from src.models.deanet.wrapper import DEANetModel
from src.models.dehazeformer.wrapper import DehazeFormerModel


class ImageRestorationAgent(BaseAgent):
    def __init__(self):
        self.deanet = None
        self.dehazeformer = None

    def _get_deanet(self):
        if self.deanet is None:
            self.deanet = DEANetModel(
            "weights/deanet/OTS/PSNR3659_SSIM9897.pth"
        )
            self.deanet.load_model()

            return self.deanet


    def _get_dehazeformer(self):
        if self.dehazeformer is None:
            self.dehazeformer = DehazeFormerModel(
            "weights/dehazeformer/outdoor/dehazeformer-t.pth"
        )
            self.dehazeformer.load_model()

            return self.dehazeformer

    def restore(self, image, strategy: str):
        """
        Executes the selected restoration strategy on the input image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.size == 0:
            raise ValueError("Input image cannot be empty.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (H, W, 3) image.")
        if image.dtype != np.uint8:
            raise TypeError("Input image must have dtype uint8.")

        strategy = strategy.lower()
        if torch.cuda.is_available() and strategy in ("deanet", "dehazeformer"):
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        if strategy == "bypass":
            restored_image = image

        elif strategy == "deanet":
            restored_image = self._get_deanet().infer(image)

        elif strategy == "dehazeformer":
            restored_image = self._get_dehazeformer().infer(image)

        else:
            raise ValueError(
        f"Unknown restoration strategy: {strategy}"
    )

        if torch.cuda.is_available() and strategy in ("deanet", "dehazeformer"):
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        latency_seconds = end_time - start_time
        latency_ms = latency_seconds * 1000

        return restored_image, latency_ms