# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch object detection project implementing a RetinaNet-style detector with Feature Pyramid Network (FPN). The project is structured as a course assignment focusing on object detection with vehicle registration plates and pedestrian datasets.

## Running the Project

### Main Entry Points
```bash
# Run experiment script
python 5-Experiment.py

# Or use Jupyter notebooks
jupyter notebook w10c1_Project3_object_detection.ipynb
```

### Training
Training is configured via dataclasses in `src/trainer/configuration.py`. The `Experiment` class in notebooks/scripts orchestrates:
- Dataset loading with `ListDataset`
- Model creation (`Detector`)
- Training via `Trainer.fit(epochs)`

## Architecture

### Model (`src/models/`)
- **Detector**: Main detection model with classification and localization heads (9 anchors per location)
- **FPN**: Feature Pyramid Network using ResNet18 backbone, outputs 5 feature maps at different scales
- **DetectionLoss**: Combined loss with SmoothL1 for localization and OHEM (Online Hard Example Mining) for classification

### Training Framework (`src/trainer/`)
- **Trainer**: Generic training loop with hooks system for customization
- **hooks.py**: Contains `train_hook_detection`, `test_hook_detection`, `end_epoch_hook_detection` for detection-specific training
- **DataEncoder**: Handles anchor box generation, encoding/decoding of predictions, NMS
- **ListDataset**: Loads images and labels from `data/{train,validation}/` directories
- **APEstimator/VOCEvaluator**: Computes mAP metric

### Key Data Flow
1. Images loaded via `ListDataset` with Albumentations transforms
2. `DataEncoder` generates anchor boxes and encodes ground truth
3. `Detector` outputs `(loc_preds, cls_preds)` tensors
4. `DetectionLoss` computes localization loss (SmoothL1) + classification loss (cross-entropy with OHEM)
5. `DataEncoder.decode()` converts predictions to bounding boxes with NMS

## Data Structure

```
data/
  train/
    Vehicle registration plate/
      *.jpg           # Training images
      Label/*.txt     # Bounding box annotations (per image)
  validation/
    Vehicle registration plate/
      *.jpg           # Validation images
annotations/
  *.json              # COCO format annotations for evaluation
```

Label files use format: `class_id confidence xmin ymin xmax ymax`

## Configuration Dataclasses

- `SystemConfig`: seed, cudnn settings
- `DatasetConfig`: root_dir, transforms
- `DataloaderConfig`: batch_size, num_workers
- `OptimizerConfig`: learning_rate, momentum, weight_decay, lr_step_milestones
- `TrainerConfig`: device, epoch_num, model_dir, progress_bar

## Key Dependencies
- PyTorch, torchvision (ResNet18 backbone)
- Albumentations (data augmentation)
- OpenCV (image processing)
- wandb (experiment tracking, optional)
- pycocotools (COCO evaluation)

## Checkpoints
Models are saved to `checkpoints/` directory. Best model saved as `Detector_best.pth`.

## Inference Utilities

`src/trainer/utils.py` provides two visualization functions:
- **`draw_bboxes()`**: Uses OpenCV rectangles, takes `TrainerConfig`
- **`run_inference()`**: Uses matplotlib patches, takes `InferenceConfig`, supports configurable `cls_threshold` and `nms_threshold` parameters. Accepts either an int (for random samples) or list of image indices.
