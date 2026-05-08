# Helmet Detection YOLOv8

This repository contains a full pipeline for training a YOLOv8 object detection model to detect whether people are wearing helmets or not (`helmet` vs `no_helmet`).

## Pipeline Features

1. **Dataset Preprocessing (`01_preprocess_dataset.py`)**
   - Validates annotation integrity across multiple YOLO datasets.
   - Detects and removes corrupted or duplicate images.
   - Merges multiple datasets into a single `combined_dataset` structure.
   - Automatically generates a `combined_data.yaml` config file.

2. **Data Augmentation & Rebalancing (`02_augment_rebalance.py`)**
   - Analyzes class distribution to identify minority classes.
   - Applies offline augmentations (flips, rotations, scaling, color jitter) to minority class images.
   - Generates new image/label pairs to balance the dataset.

3. **YOLOv8 Training (`03_train_yolov8.py`)**
   - Uses Ultralytics YOLOv8 for object detection.
   - Implements advanced augmentations (mosaic, mixup, fliplr).
   - Configured with Cosine LR scheduling, AdamW optimizer, and early stopping.
   - Automatically detects CPU vs. GPU and optimizes settings.

4. **Model Evaluation (`04_evaluate_model.py`)**
   - Evaluates the trained model on validation and test splits.
   - Generates key metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall, and F1-score.
   - Produces a confusion matrix and an error analysis report for misclassifications.

5. **Model Export (`05_export_model.py`)**
   - Exports the best weights to the `models/` directory.
   - Converts the model to `ONNX` and `TorchScript` formats for deployment.
   - Includes a brief inference speed benchmark to ensure real-time readiness.

## How to Run

Install dependencies:
```bash
pip install ultralytics Pillow matplotlib numpy pyyaml onnx onnxruntime
```

Run the pipeline sequentially:
```bash
python 01_preprocess_dataset.py
python 02_augment_rebalance.py
python 03_train_yolov8.py
python 04_evaluate_model.py
python 05_export_model.py
```