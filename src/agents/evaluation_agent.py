from src.agents.base_agent import BaseAgent

class EvaluationAgent(BaseAgent):
    def evaluate(self, results: dict) -> dict:
        """
        Evaluates perception and computational performance.
        """
        raise NotImplementedError('Evaluation is not yet implemented.')
