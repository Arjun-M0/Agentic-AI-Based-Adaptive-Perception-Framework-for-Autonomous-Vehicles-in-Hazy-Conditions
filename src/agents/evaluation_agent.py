from src.agents.base_agent import BaseAgent


class EvaluationAgent(BaseAgent):

    def evaluate(self, results: dict) -> dict:
        """
        Evaluates detection and computational performance.
        """

        evaluation = {
            "mAP": results.get("mAP"),
            "precision": results.get("precision"),
            "recall": results.get("recall"),
            "latency_ms": results.get("latency_ms"),
            "fps": results.get("fps")
        }

        return evaluation