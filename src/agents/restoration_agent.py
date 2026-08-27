from src.agents.base_agent import BaseAgent

class ImageRestorationAgent(BaseAgent):
    def restore(self, image, strategy: str):
        """
        Executes the given restoration strategy (model inference) on the input image.
        """
        raise NotImplementedError('Image restoration is not yet implemented.')
