import time
from src.agents.base_agent import BaseAgent
from src.models.deanet.wrapper import DEANetModel


class ImageRestorationAgent(BaseAgent):
    def __init__(self):
        self.deanet = DEANetModel(
            "weights/deanet/OTS/PSNR3659_SSIM9897.pth"
        )

        self.deanet.load_model()

    def restore(self, image, strategy: str):
        """
        Executes the selected restoration strategy on the input image.
        """
        strategy = strategy.lower()

        start_time = time.perf_counter()

        if strategy == "bypass":
            restored_image = image

        elif strategy == "deanet":
            restored_image = self.deanet.infer(image)

        else:
            raise ValueError(
            f"Unknown restoration strategy: {strategy}"
        )

        end_time = time.perf_counter()

        latency_seconds = end_time - start_time
        latency_ms = latency_seconds * 1000

        return restored_image, latency_ms