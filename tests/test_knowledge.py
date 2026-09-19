import sys
import os
import json
import tempfile
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.knowledge_agent import KnowledgeRepositoryAgent
from src.core.context import EnvironmentState, HardwareState

@pytest.fixture
def temp_repo_path():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(b"[]")
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_repo_initialization(temp_repo_path):
    agent = KnowledgeRepositoryAgent(file_path=temp_repo_path)
    assert agent.get_all() == []

def test_add_experience(temp_repo_path):
    agent = KnowledgeRepositoryAgent(file_path=temp_repo_path)
    exp = {"strategy": "Bypass", "outcome": {"success": True}}
    assert agent.update(exp) == True
    
    history = agent.get_all()
    assert len(history) == 1
    assert "experience_id" in history[0]
    assert "timestamp" in history[0]
    assert history[0]["strategy"] == "Bypass"

def test_multiple_experiences(temp_repo_path):
    agent = KnowledgeRepositoryAgent(file_path=temp_repo_path)
    agent.update({"strategy": "Bypass"})
    agent.update({"strategy": "DEA-Net"})
    assert len(agent.get_all()) == 2

def test_retrieval(temp_repo_path):
    agent = KnowledgeRepositoryAgent(file_path=temp_repo_path)
    # Add experiences
    agent.update({
        "environment": {"haze_score": 0.2, "visibility_score": 0.8},
        "hardware": {"gpu_available": False},
        "strategy": "Bypass"
    })
    agent.update({
        "environment": {"haze_score": 0.9, "visibility_score": 0.1},
        "hardware": {"gpu_available": True},
        "strategy": "DehazeFormer"
    })
    
    env = EnvironmentState(haze_score=0.9, visibility_score=0.1)
    hw = HardwareState(gpu_available=True)
    
    retrieved = agent.retrieve(environment_state=env, hardware_state=hw)
    assert len(retrieved) > 0
    assert retrieved[0]["strategy"] == "DehazeFormer"

def test_top_k(temp_repo_path):
    agent = KnowledgeRepositoryAgent(file_path=temp_repo_path, top_k=2)
    for i in range(5):
        agent.update({"strategy": f"Strategy_{i}"})
    assert len(agent.retrieve()) == 2

def test_missing_file():
    path = "non_existent_dir/test_history.json"
    if os.path.exists(path):
        os.remove(path)
    agent = KnowledgeRepositoryAgent(file_path=path)
    assert agent.get_all() == []
    assert agent.retrieve() == []

def test_malformed_json(temp_repo_path):
    with open(temp_repo_path, "w", encoding="utf-8") as f:
        f.write("{malformed json")
    agent = KnowledgeRepositoryAgent(file_path=temp_repo_path)
    assert agent.get_all() == []

def test_reset(temp_repo_path):
    agent = KnowledgeRepositoryAgent(file_path=temp_repo_path)
    agent.update({"strategy": "DEA-Net"})
    assert len(agent.get_all()) == 1
    agent.reset()
    assert len(agent.get_all()) == 0
