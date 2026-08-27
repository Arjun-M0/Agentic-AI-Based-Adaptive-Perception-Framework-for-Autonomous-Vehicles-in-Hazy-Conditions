from src.agents.base_agent import BaseAgent
from src.core.context import EnvironmentState

class EnvironmentUnderstandingAgent(BaseAgent):
    def analyze(self, image) -> EnvironmentState:
        """
        Analyzes the environmental/visual condition of the current frame.
        Returns an EnvironmentState containing haze, visibility, and illumination scores.
        """
        raise NotImplementedError('Environment analysis is not yet implemented.')
