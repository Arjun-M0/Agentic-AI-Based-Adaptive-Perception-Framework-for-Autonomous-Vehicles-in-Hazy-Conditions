import json
import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.agents.base_agent import BaseAgent
from src.core.context import EnvironmentState, HardwareState

logger = logging.getLogger(__name__)

class KnowledgeRepositoryAgent(BaseAgent):
    def __init__(self, file_path: str = "knowledge/restoration_history.json", top_k: int = 5):
        self.file_path = file_path
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
                if isinstance(data, list):
                    return data
                return []
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading knowledge repository: {e}")
            return []

    def _save_history(self, history: List[Dict[str, Any]]) -> bool:
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            temp_path = self.file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
            os.replace(temp_path, self.file_path)
            return True
        except IOError as e:
            logger.error(f"Error writing to knowledge repository: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
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

            scored_history.append((score, exp))

        scored_history.sort(key=lambda x: x[0], reverse=True)
        return [exp for score, exp in scored_history[:self.top_k]]

    def update(self, experience: Dict[str, Any]) -> bool:
        """
        Updates the historical repository with a new experience.
        """
        history = self._load_history()
        
        if "experience_id" not in experience:
            experience["experience_id"] = f"exp_{uuid.uuid4().hex[:8]}"
        if "timestamp" not in experience:
            experience["timestamp"] = datetime.utcnow().isoformat() + "Z"
            
        history.append(experience)
        return self._save_history(history)
        
    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all experiences."""
        return self._load_history()
        
    def reset(self) -> bool:
        """Clears the repository. Useful for testing."""
        return self._save_history([])
