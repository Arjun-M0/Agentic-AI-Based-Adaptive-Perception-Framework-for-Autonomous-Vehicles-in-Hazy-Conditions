import json
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from src.agents.base_agent import BaseAgent
from src.core.context import EnvironmentState, HardwareState

logger = logging.getLogger(__name__)

class KnowledgeRepositoryAgent(BaseAgent):
    def __init__(self, file_path: str = "knowledge/restoration_history.json", top_k: int = 5):
        self.file_path = file_path
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.top_k = top_k

    def _load_history(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    return []
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("knowledge repository must contain a JSON list")
                experiences = [item for item in data if isinstance(item, dict)]
                if len(experiences) != len(data):
                    logger.error("Ignoring non-object entries in knowledge repository")
                return experiences
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.error(f"Error reading knowledge repository: {e}")
            return []

    def _save_history(self, history: List[Dict[str, Any]]) -> bool:
        try:
            directory = os.path.dirname(self.file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            temp_path = self.file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
            os.replace(temp_path, self.file_path)
            return True
        except OSError as e:
            logger.error(f"Error writing to knowledge repository: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    logger.error("Could not remove temporary knowledge repository file")
            return False

    def retrieve(self, environment_state: Optional[EnvironmentState] = None, hardware_state: Optional[HardwareState] = None) -> List[Dict[str, Any]]:
        """
        Retrieves historical experiences similar to the current state.
        Uses a basic similarity metric based on environment and hardware.
        """
        history = self._load_history()
        if not history:
            return []

        if environment_state is None and hardware_state is None:
            return sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:self.top_k]

        scored_history = []
        for exp in history:
            score = 0.0
            
            exp_env = exp.get("environment", {})
            if environment_state:
                if environment_state.haze_score is not None and "haze_score" in exp_env:
                    score += max(0.0, 1.0 - abs(environment_state.haze_score - exp_env["haze_score"]))
                if environment_state.visibility_score is not None and "visibility_score" in exp_env:
                    score += max(0.0, 1.0 - abs(environment_state.visibility_score - exp_env["visibility_score"]))

            exp_hw = exp.get("hardware", {})
            if hardware_state:
                if "gpu_available" in exp_hw and hardware_state.gpu_available == exp_hw["gpu_available"]:
                    score += 1.0

            scored_history.append((score, exp.get("timestamp", ""), exp))

        scored_history.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [exp for _, _, exp in scored_history[:self.top_k]]

    def has_similar_experience(
        self,
        environment_state: Optional[EnvironmentState],
        hardware_state: Optional[HardwareState],
        threshold: float = 0.2,
    ) -> bool:
        """Return whether history contains a usable experience for this state."""
        for experience in self._load_history():
            environment = experience.get("environment", {})
            if not isinstance(environment, dict):
                continue
            environment_matches = [
                abs(environment[field] - value) <= threshold
                for field, value in {
                    "haze_score": getattr(environment_state, "haze_score", None),
                    "visibility_score": getattr(environment_state, "visibility_score", None),
                    "illumination_score": getattr(environment_state, "illumination_score", None),
                    "contrast_score": getattr(environment_state, "contrast_score", None),
                }.items()
                if value is not None and isinstance(environment.get(field), (int, float))
            ]
            hardware = experience.get("hardware", {})
            hardware_matches = (
                hardware_state is not None
                and isinstance(hardware, dict)
                and isinstance(hardware.get("gpu_available"), bool)
                and hardware.get("gpu_available") == hardware_state.gpu_available
            )
            has_hardware_data = (
                isinstance(hardware, dict) and "gpu_available" in hardware
            )
            environment_matches_state = (
                bool(environment_matches) and all(environment_matches)
            )
            if environment_matches_state and (
                not has_hardware_data or hardware_matches
            ):
                return True
            if not environment_matches and hardware_matches:
                return True
        return False

    @staticmethod
    def best_strategy(experiences: List[Dict[str, Any]]) -> Optional[str]:
        """Return the fastest successful strategy in retrieved experiences."""
        candidates = []
        for experience in experiences:
            if experience.get("outcome", {}).get("success") is not True:
                continue
            strategy = experience.get("strategy")
            latency = experience.get("performance", {}).get("latency_ms")
            if isinstance(strategy, str) and isinstance(latency, (int, float)):
                candidates.append((float(latency), strategy))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def update(self, experience: Dict[str, Any]) -> bool:
        """
        Updates the historical repository with a new experience.
        """
        if not isinstance(experience, dict):
            raise TypeError("experience must be a dictionary")
        history = self._load_history()

        experience = dict(experience)
        if "experience_id" not in experience:
            experience["experience_id"] = f"exp_{uuid.uuid4().hex[:8]}"
        if "timestamp" not in experience:
            experience["timestamp"] = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            
        history.append(experience)
        return self._save_history(history)
        
    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all experiences."""
        return self._load_history()
        
    def reset(self) -> bool:
        """Clears the repository. Useful for testing."""
        return self._save_history([])
