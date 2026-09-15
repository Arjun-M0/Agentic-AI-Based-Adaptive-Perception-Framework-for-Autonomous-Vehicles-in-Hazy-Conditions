from src.agents.restoration_agent import ImageRestorationAgent


def test_bypass_returns_same_image():
    agent = ImageRestorationAgent()

    sample_image = "dummy_image"

    result = agent.restore(sample_image, "bypass")

    assert result == sample_image