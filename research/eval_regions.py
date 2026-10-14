"""
eval_regions.py: Coordinate Transformation and Region Evaluation Metrics for Coord-OPD

Handles:
1. Transforming raw action offsets into valid pixel/normalized crop bounding boxes:
   b_final = clip(b_coarse + scale * z, [0, 0, 1, 1])
2. Detecting degenerate, inverted, or out-of-bounds crops.
3. Calculating Intersection over Union (IoU) and coverage on evidence regions.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def apply_action_transform(
    coarse_box: List[float],
    raw_action: List[float],
    scale: float = 0.2,
) -> List[float]:
    """
    Applies continuous action offset to coarse bounding box.
    Boxes are [x1, y1, x2, y2] normalized in [0, 1].
    
    Args:
        coarse_box: [x1, y1, x2, y2]
        raw_action: [dx1, dy1, dx2, dy2] continuous offsets
        scale: Scale factor for raw action residuals
        
    Returns:
        final_box: [x1, y1, x2, y2] clamped to [0, 1]
    """
    x1 = coarse_box[0] + raw_action[0] * scale
    y1 = coarse_box[1] + raw_action[1] * scale
    x2 = coarse_box[2] + raw_action[2] * scale
    y2 = coarse_box[3] + raw_action[3] * scale
    
    # Ensure min < max
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # Clamp to [0, 1]
    return [
        max(0.0, min(1.0, x_min)),
        max(0.0, min(1.0, y_min)),
        max(0.0, min(1.0, x_max)),
        max(0.0, min(1.0, y_max)),
    ]


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 1e-8:
        return 0.0
    return inter_area / union_area


def is_valid_box(box: List[float], min_side: float = 0.01) -> bool:
    """Checks if box has positive area and meets minimum side length."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w >= min_side and h >= min_side


if __name__ == "__main__":
    print("Running self-check for eval_regions.py...")
    coarse = [0.2, 0.2, 0.6, 0.6]
    action = [0.1, -0.1, 0.2, 0.1]
    final = apply_action_transform(coarse, action, scale=0.2)
    
    assert is_valid_box(final), "Box should be valid!"
    iou_self = compute_box_iou(final, final)
    assert abs(iou_self - 1.0) < 1e-5, f"Self-IoU should be 1.0, got {iou_self}"
    
    gt_box = [0.22, 0.18, 0.65, 0.62]
    iou = compute_box_iou(final, gt_box)
    assert iou > 0.8, f"Expected high IoU, got {iou}"
    print(f"Action transform check passed! Coarse: {coarse} -> Final: {[round(x, 3) for x in final]} (IoU: {iou:.3f})")
    print("All unit tests in eval_regions.py passed successfully!")
