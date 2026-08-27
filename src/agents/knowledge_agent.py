from src.agents.base_agent import BaseAgent

class KnowledgeRepositoryAgent(BaseAgent):
    def retrieve(self, current_state: dict) -> dict:
        """
        Retrieves historical experiences similar to the current state.
        """
        raise NotImplementedError('Knowledge retrieval is not yet implemented.')

    def update(self, experience: dict):
        """
        Updates the historical repository with a new experience.
        """
        raise NotImplementedError('Knowledge update is not yet implemented.')
