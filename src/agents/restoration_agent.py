from src.agents.base_agent import BaseAgent


class ImageRestorationAgent(BaseAgent):
    def restore(self, image, strategy: str):
        """
        Executes the given restoration strategy on the input image.
        """

        strategy = strategy.lower()

        if strategy == "bypass":
            return image

        raise ValueError(f"Unknown restoration strategy: {strategy}")