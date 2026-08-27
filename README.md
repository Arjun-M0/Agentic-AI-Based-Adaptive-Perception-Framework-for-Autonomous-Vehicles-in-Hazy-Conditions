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

## 4. Current Project Status
Status: **Initial Backbone / Skeleton** (Placeholder interfaces implemented).
