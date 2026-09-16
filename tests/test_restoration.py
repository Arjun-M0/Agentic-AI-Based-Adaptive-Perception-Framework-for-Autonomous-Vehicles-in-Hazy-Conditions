from src.agents.restoration_agent import ImageRestorationAgent
import numpy as np


def test_deanet_returns_image_with_same_shape():
    agent = ImageRestorationAgent()

    sample_image = np.zeros((100, 200, 3), dtype=np.uint8)

    result, latency = agent.restore(sample_image, "deanet")

    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    assert latency >= 0


def test_bypass_returns_same_image():
    agent = ImageRestorationAgent()

    sample_image = "dummy_image"

    result, latency = agent.restore(sample_image, "bypass")

    assert result == sample_image
    assert latency >= 0

def test_dehazeformer_returns_image_with_same_shape():
    agent = ImageRestorationAgent()

    sample_image = np.zeros((100, 200, 3), dtype=np.uint8)

    result, latency = agent.restore(sample_image, "dehazeformer")

    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    assert latency >= 0