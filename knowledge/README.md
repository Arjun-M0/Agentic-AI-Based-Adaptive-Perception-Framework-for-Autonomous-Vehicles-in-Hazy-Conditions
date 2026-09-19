# Knowledge Repository

This folder stores historical restoration experiences for the Agentic AI-Based Adaptive Perception Framework.
It allows the Adaptive Decision Agent to learn from past performance in varying environmental and hardware conditions.

## Schema
Experiences are stored in `restoration_history.json` as a list of JSON objects:

```json
[
  {
    "experience_id": "exp_1694500123",
    "timestamp": "2023-09-12T10:02:03Z",
    "environment": {
      "haze_score": 0.8,
      "visibility_score": 0.2
    },
    "hardware": {
      "gpu_available": true,
      "gpu_memory_available": 4096,
      "target_fps": 30
    },
    "strategy": "DEA-Net",
    "requirements": {
      "target_fps": 30.0,
      "max_latency_ms": 100.0
    },
    "performance": {
      "fps": 28.5,
      "latency_ms": 35.0,
      "map": 0.75
    },
    "outcome": {
      "success": true
    },
    "notes": "Good balance of speed and performance."
  }
]
```

## Usage
- **Storage**: JSON is used as a lightweight local store. The agent reads and appends to this file.
- **Cold Start**: If the repository is empty, the Adaptive Decision Agent relies purely on current hardware/environment factors and default configurations, rather than past experiences.
- **Test Fixtures**: Any synthetic data used for unit testing will be kept strictly in `tests/fixtures/mock_experiences.json`. This repository should only contain real experimental data.

**Developer Note**:
Current implementation:
- uses local JSON storage
- uses an explainable multi-criteria decision mechanism
- does not claim autonomous learning unless real feedback is incorporated
