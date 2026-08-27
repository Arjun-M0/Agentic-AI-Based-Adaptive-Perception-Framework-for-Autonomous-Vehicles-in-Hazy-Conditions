from src.agents.base_agent import BaseAgent
from src.core.context import HardwareState

class HardwareAwarenessAgent(BaseAgent):
    def analyze(self) -> HardwareState:
        """
        Analyzes the current computational environment (GPU, CPU, RAM).
        Returns a HardwareState containing current resource utilization.
        """
        raise NotImplementedError('Hardware monitoring is not yet implemented.')
