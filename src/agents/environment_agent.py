import logging
from src.agents.base_agent import BaseAgent
from src.core.context import EnvironmentState
from src.utils.image_utils import (
    calculate_haze_score,
    calculate_visibility_score,
    calculate_illumination,
    calculate_contrast
)

logger = logging.getLogger(__name__)

class EnvironmentUnderstandingAgent(BaseAgent):
    """
    Agent responsible for analyzing the visual environment of the driving frame.
    Calculates empirical metrics for haze, visibility, illumination, and contrast.
    """
    
    def analyze(self, image) -> EnvironmentState:
        """
        Analyzes the environmental/visual condition of the current frame.
        Returns an EnvironmentState containing structural and illumination metrics.
        """
        state = EnvironmentState()
        
        if image is None or image.size == 0:
            logger.warning("Received empty image for environment analysis.")
            return state
            
        try:
            # 1. Illumination (Brightness) & Contrast
            state.illumination_score = calculate_illumination(image)
            state.contrast_score = calculate_contrast(image)
            
            # 2. Haze Severity & Visibility
            state.haze_score = calculate_haze_score(image)
            state.visibility_score = calculate_visibility_score(image)
            
        except Exception as e:
            logger.error(f"Error during environment analysis: {e}")
            
        return state
