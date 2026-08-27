from src.agents.base_agent import BaseAgent
from src.core.context import EnvironmentState, HardwareState

class AdaptiveDecisionAgent(BaseAgent):
    def select_strategy(self, environment_state: EnvironmentState, hardware_state: HardwareState, historical_knowledge: dict) -> str:
        """
        Selects the best restoration strategy based on state and historical knowledge.
        Returns the selected strategy (e.g., 'Bypass', 'DEA-Net', 'DehazeFormer').
        """
        raise NotImplementedError('Adaptive strategy selection is not yet implemented.')
