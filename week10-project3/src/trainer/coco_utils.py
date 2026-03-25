import os
import json
import cv2
import torch
from tqdm.auto import tqdm
from .encoder import DataEncoder


def generate_coco_detection_file(
    exp,
    image_ids=None,
    cls_threshold=0.3,
    nms_threshold=0.3,
    output_path="cocoResults",
    output_filename="detections.json"
):
    """
    Generate COCO format detection results file and ground truth annotations file.

    Args:
        exp: Experiment object containing model, dataset_test, loader_test, and inference_config
        image_ids: Optional list of dataset indices to process. If None, processes all images.
        cls_threshold: Classification confidence threshold for detections (default: 0.3)
        nms_threshold: NMS IoU threshold (default: 0.3)
        output_path: Directory to save the output files (default: "cocoResults")
        output_filename: Name of the detection JSON file (default: "detections.json")

    Returns:
        tuple: (detection_file_path, ground_truth_file_path)

    Output formats:
        Detection file (COCO detection results):
            [{"image_id": int, "category_id": 1, "bbox": [x, y, width, height], "score": float}, ...]

        Ground truth file (COCO annotations):
            {"info": {...}, "images": [...], "categories": [...], "annotations": [...]}
    """
    dataset = exp.loader_test.dataset

    # Default to all images if image_ids not provided
    if image_ids is None:
        image_ids = list(range(len(dataset)))

    print(f"Processing {len(image_ids)} images...")

    input_size = exp.dataset_test.input_size
    encoder = DataEncoder((input_size, input_size))
    device = exp.inference_config.device

    # Ensure model is in eval mode
    exp.model.eval()

    all_detections = []

    # Ground truth structures
    gt_images = []
    gt_annotations = []
    annotation_id = 1  # Unique ID for each annotation

    with torch.no_grad():
        for index in tqdm(image_ids, desc="Generating detections"):
            # Get original image to obtain dimensions
            img_path = os.path.join(dataset.root_dir, dataset.fnames[index])
            orig_img = cv2.imread(img_path)
            orig_height, orig_width = orig_img.shape[:2]

            # Calculate scale factors to convert from input_size back to original
            scale_w = orig_width / input_size
            scale_h = orig_height / input_size

            # Add image entry to ground truth
            gt_images.append({
                "license": 1,
                "file_name": dataset.fnames[index],
                "coco_url": "",
                "height": orig_height,
                "width": orig_width,
                "date_captured": "2026-03-01 00:00:00",
                "id": int(index)
            })

            # Add ground truth annotations (original coordinates from dataset.boxes)
            gt_boxes = dataset.boxes[index]  # Original coordinates
            for box in gt_boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                box_width = xmax - xmin
                box_height = ymax - ymin
                gt_annotations.append({
                    "id": annotation_id,
                    "image_id": int(index),
                    "category_id": 1,
                    "bbox": [
                        round(xmin, 2),
                        round(ymin, 2),
                        round(box_width, 2),
                        round(box_height, 2)
                    ],
                    "area": round(box_width * box_height, 2),
                    "iscrowd": 0
                })
                annotation_id += 1

            # Get model predictions
            image, _, _, _ = dataset[index]
            image = image.to(device).clone()
            loc_preds, cls_preds = exp.model(image.unsqueeze(0))

            samples = encoder.decode(
                loc_preds,
                cls_preds,
                cls_threshold=cls_threshold,
                nms_threshold=nms_threshold
            )

            # samples[0][1] contains detections for class 1 (Reg_Plate)
            # Each row: [xmin, ymin, xmax, ymax, confidence]
            if samples[0][1].size > 0:
                for detection in samples[0][1]:
                    xmin, ymin, xmax, ymax, confidence = detection

                    # Scale from input_size coordinates back to original image coordinates
                    xmin_orig = xmin * scale_w
                    ymin_orig = ymin * scale_h
                    xmax_orig = xmax * scale_w
                    ymax_orig = ymax * scale_h

                    # Convert to COCO format [x, y, width, height]
                    x = float(xmin_orig)
                    y = float(ymin_orig)
                    width = float(xmax_orig - xmin_orig)
                    height = float(ymax_orig - ymin_orig)

                    all_detections.append({
                        "image_id": int(index),
                        "category_id": 1,
                        "bbox": [round(x, 2), round(y, 2), round(width, 2), round(height, 2)],
                        "score": round(float(confidence), 3)
                    })

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Write detections to JSON file
    detection_file = os.path.join(output_path, output_filename)
    with open(detection_file, 'w') as f:
        json.dump(all_detections, f)

    # Build ground truth annotations structure
    gt_data = {
        "info": {
            "description": "Ground truth annotations for object detection",
            "version": "1.0",
            "year": 2026,
            "date_created": "2026-03-01"
        },
        "images": gt_images,
        "categories": [
            {"supercategory": "Reg_Plate", "id": 1, "name": "Reg_Plate"}
        ],
        "annotations": gt_annotations
    }

    # Derive ground truth filename from detection filename
    name, ext = os.path.splitext(output_filename)
    gt_filename = f"{name}_gt{ext}"
    gt_file = os.path.join(output_path, gt_filename)

    with open(gt_file, 'w') as f:
        json.dump(gt_data, f)

    print(f"Generated {len(all_detections)} detections from {len(image_ids)} images")
    print(f"Generated {len(gt_annotations)} ground truth annotations")
    print(f"Detection file: {detection_file}")
    print(f"Ground truth file: {gt_file}")

    return detection_file, gt_file
