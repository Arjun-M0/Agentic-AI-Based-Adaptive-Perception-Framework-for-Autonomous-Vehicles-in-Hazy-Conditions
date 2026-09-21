import os
import pytest
import numpy as np
from src.agents.restoration_agent import ImageRestorationAgent


DEANET_WEIGHTS_EXIST = os.path.exists("weights/deanet/OTS/PSNR3659_SSIM9897.pth")
DEHAZEFORMER_WEIGHTS_EXIST = os.path.exists("weights/dehazeformer/outdoor/dehazeformer-t.pth")


def test_bypass_returns_same_image():
    agent = ImageRestorationAgent()
    sample_image = np.zeros((100, 200, 3), dtype=np.uint8)
    
    result, latency = agent.restore(sample_image, "bypass")
    
    np.testing.assert_array_equal(result, sample_image)
    assert latency >= 0


def test_invalid_strategy():
    agent = ImageRestorationAgent()
    sample_image = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown restoration strategy"):
        agent.restore(sample_image, "invalid_strategy")


def test_invalid_input_type():
    agent = ImageRestorationAgent()
    with pytest.raises(TypeError, match="Input image must be a numpy array"):
        agent.restore("not_an_image", "bypass")


def test_invalid_input_shape():
    agent = ImageRestorationAgent()
    sample_image = np.zeros((100, 200), dtype=np.uint8)  # Grayscale
    with pytest.raises(ValueError, match="Input image must be a 3-channel"):
        agent.restore(sample_image, "bypass")


def test_invalid_input_dtype():
    agent = ImageRestorationAgent()
    sample_image = np.zeros((100, 200, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="Input image must have dtype uint8"):
        agent.restore(sample_image, "bypass")


def test_empty_input_image():
    agent = ImageRestorationAgent()
    sample_image = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="Input image cannot be empty"):
        agent.restore(sample_image, "bypass")


@pytest.mark.skipif(not DEANET_WEIGHTS_EXIST, reason="DEA-Net weights not found")
def test_deanet_returns_image_with_same_shape():
    agent = ImageRestorationAgent()
    sample_image = np.zeros((100, 200, 3), dtype=np.uint8)

    result, latency = agent.restore(sample_image, "deanet")

    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    assert latency >= 0


@pytest.mark.skipif(not DEHAZEFORMER_WEIGHTS_EXIST, reason="DehazeFormer weights not found")
def test_dehazeformer_returns_image_with_same_shape():
    agent = ImageRestorationAgent()
    sample_image = np.zeros((100, 200, 3), dtype=np.uint8)

    result, latency = agent.restore(sample_image, "dehazeformer")

    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    assert latency >= 0