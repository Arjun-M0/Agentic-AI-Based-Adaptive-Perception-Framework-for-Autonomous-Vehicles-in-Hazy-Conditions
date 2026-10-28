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
- **Storage**: JSON is used as a lightweight local store. The agent reads and appends to this file atomically.
- **Feedback**: Each successfully processed pipeline frame appends the measured
  environment, hardware, strategy, restoration/detection latency, FPS, and provenance.
  Retrieval happens before the next decision, allowing later frames to use measured
  outcomes.
  - **Latency evidence**: A cold-start experience stores the measured total latency
    of the Bypass, DEA-Net, and DehazeFormer candidate runs. A later similar
    experience stores the measured single-strategy latency and the calculated
    reduction relative to that cold-start exploration total. The comparison is
    available in the pipeline result and Excel report.
- **Cold Start**: If the repository is empty, the Adaptive Decision Agent relies purely on current hardware/environment factors and default configurations, rather than past experiences.
- **Test Fixtures**: Any synthetic data used for unit testing will be kept strictly in `tests/fixtures/mock_experiences.json`. This repository should only contain real experimental data.

**Developer Note**:
Current implementation:
- uses local JSON storage
- uses an explainable multi-criteria decision mechanism
- incorporates measured pipeline feedback but does not claim autonomous model training

## Latency comparison

Run a same-image lifecycle comparison:

```powershell
python scripts\benchmark_knowledge_latency.py
```

In the actual `PerceptionPipeline`, a frame with no similar experience enters
cold-start exploration: each configured strategy is executed once, measured by
the detector, and compared under the decision agent's constraints. Only the
selected strategy's experience is persisted for future routing. A later frame
with a similar environment/hardware state retrieves that experience and executes
only the selected strategy; the other restoration models are not run.

The standalone script is only an optional measurement utility. It must not be
treated as the normal pipeline implementation or as evidence that all three
models run for every frame.

## Demonstrating the latency reduction

Use an isolated repository for a clean teacher demonstration. This prevents
existing experiences in `knowledge/restoration_history.json` from affecting the
first run:

```powershell
New-Item -ItemType Directory -Force outputs\reports\knowledge_demo, outputs\detections\knowledge_demo | Out-Null
Remove-Item outputs\reports\knowledge_demo\demo_history.json -ErrorAction SilentlyContinue

# Run 1: no similar knowledge; explores all three strategies.
python main.py data\samples\test_road.jpg `
  --knowledge outputs\reports\knowledge_demo\demo_history.json `
  --report outputs\reports\knowledge_demo\before_knowledge.xlsx `
  --detections outputs\detections\knowledge_demo\before

# Run 2: same image; retrieves the stored experience and runs only the selected strategy.
python main.py data\samples\test_road.jpg `
  --knowledge outputs\reports\knowledge_demo\demo_history.json `
  --report outputs\reports\knowledge_demo\after_knowledge.xlsx `
  --detections outputs\detections\knowledge_demo\after
```

Show the following files:

- `demo_history.json`: actual candidate and selected-strategy latency records.
- `before_knowledge.xlsx`: `Cold-Start Exploration = True` and
  `Candidate Strategies Evaluated = 3`.
- `after_knowledge.xlsx`: `Cold-Start Exploration = False`,
  `Candidate Strategies Evaluated = 1`, and `Latency Reduction (%)`.

The second report compares the one-strategy latency with the stored total
latency of the first run's three candidate executions. All values are measured
from restoration and detection; no latency values are fabricated.
