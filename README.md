# Agentic AI-Based Adaptive Perception Framework for Autonomous Vehicles in Hazy Conditions

## 1. Problem Statement
Autonomous vehicles rely on camera-based perception for detecting road objects. Hazy, foggy, and low-visibility conditions degrade image quality. Applying a fixed restoration method isn't optimal due to varying conditions and hardware resources.

## 2. Proposed Solution
An agentic framework that dynamically determines the optimal restoration strategy (Bypass, DEA-Net, or DehazeFormer) based on environmental conditions, computational resources, and historical knowledge, followed by YOLOv11 object detection.

## 3. Architecture
- **Input Layer**: Handles driving images/video frames.
- **Perception, Resource & Decision Layer**: Contains Environment Understanding, Hardware Awareness, Adaptive Decision, and Knowledge Repository agents.
- **Processing Layer**: Executes the selected Image Restoration strategy and performs Object Detection (YOLOv11).
- **Evaluation & Knowledge Feedback Layer**: Evaluates perception and runtime metrics, feeding results back to update historical knowledge.

## 4. Running an Evaluation

Run the complete pipeline through `main.py`:

```powershell
python main.py "data\mid_review\images" `
  --report outputs\reports\mid_review_batch_evaluation.xlsx `
  --detections outputs\detections\mid_review_batch
```

The input may be one image or a directory. Directory input is discovered
recursively and supports JPG, JPEG, PNG, BMP, and WEBP files. The Excel
workbook contains:

- `Summary`: processed/skipped counts and metric availability
- `Per Image`: environment measurements, selected strategy, explanation, and runtime measurements
- `Detections`: each YOLO detection and bounding box
- `Skipped`: files that could not be decoded
- `Strategy Summary`: aggregate latency and FPS by selected strategy

The selected 30-image dataset is unannotated, so mAP, precision, and recall
are reported as unavailable rather than fabricated. Detection count,
confidence, restoration latency, total latency, and FPS are measured from the
actual run. Historical retrieval is optional; an empty knowledge repository is
reported by the decision explanation as a cold start.

## 5. Current Project Status
Status: **End-to-end evaluation pipeline integrated**. The repository's
available model weights and YOLO detector are used directly. If a selected restoration model cannot be loaded, the run fails explicitly
instead of substituting a mock result.
