"""
eval_regions.py: Coordinate Transformation and Region Evaluation Metrics for Coord-OPD

Two transforms live here:

License / attribution: sanitize_normalized_box and the residual step in
apply_nv_cot_action_transform are adapted from kesenzhao/NV-CoT
continuous_policy.py (sanitize_normalized_boxes / apply_text_box_residual),
Apache-2.0, pinned commit 0d36f9c515fc40864c3c4497f5de73148b77b8ac.
Modification: ported from batched torch tensors to plain-Python single boxes.
The upstream license is reproduced in third_party/NV-CoT/LICENSE; the rest of
this file is original to this repository.

1. Upstream-faithful port (use this for anything that must match NV-CoT replay):
   a faithful port of the pinned upstream path (rollout.py / continuous_policy.py
   at commit 0d36f9c515fc40864c3c4497f5de73148b77b8ac):

       coarse = sanitize_normalized_boxes(text_bbox / 1000, 0.02)   # rollout.py:43
       final  = sanitize_normalized_boxes(coarse + 0.15 * tanh(raw), 0.02)  # rollout.py:49

   Implemented as text_box_to_coarse / sanitize_normalized_box /
   apply_nv_cot_action_transform. Box-level only: the upstream pixel mapping and
   resized_crop step is NOT ported (see docs/INTEGRATION_NOTES.md).

2. apply_action_transform: the original simple prototype (coarse + scale * raw,
   sort, clamp). NOT upstream-aligned; kept only as a reference prototype.
"""

import math
from typing import List


def sanitize_normalized_box(box: List[float], min_size: float = 0.02) -> List[float]:
    """Port of upstream continuous_policy.sanitize_normalized_boxes (single box).

    Center/size parameterization: keeps the box differentiable-friendly even for
    reversed coordinates, enforces min_size on each side, and shifts the box back
    into the [0, 1] frame instead of hard-sorting or clipping coordinates.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(abs(x2 - x1), min_size)
    h = max(abs(y2 - y1), min_size)
    lo_x, lo_y = cx - w * 0.5, cy - h * 0.5
    hi_x, hi_y = cx + w * 0.5, cy + h * 0.5
    shift_lo_x, shift_hi_x = max(0.0, -lo_x), max(0.0, hi_x - 1.0)
    shift_lo_y, shift_hi_y = max(0.0, -lo_y), max(0.0, hi_y - 1.0)
    lo_x = min(max(lo_x + shift_lo_x - shift_hi_x, 0.0), 1.0 - min_size)
    lo_y = min(max(lo_y + shift_lo_y - shift_hi_y, 0.0), 1.0 - min_size)
    hi_x = min(max(hi_x + shift_lo_x - shift_hi_x, min_size), 1.0)
    hi_y = min(max(hi_y + shift_lo_y - shift_hi_y, min_size), 1.0)
    hi_x = min(max(hi_x, lo_x + min_size), 1.0)
    hi_y = min(max(hi_y, lo_y + min_size), 1.0)
    return [lo_x, lo_y, hi_x, hi_y]


def text_box_to_coarse(text_box: List[float], min_size: float = 0.02) -> List[float]:
    """Upstream rollout.py:43 — text boxes are written in 0-1000 units; divide by
    1000 and sanitize into a normalized coarse box before the residual step."""
    return sanitize_normalized_box([v / 1000.0 for v in text_box], min_size)


def apply_nv_cot_action_transform(
    coarse_box: List[float],
    raw_action: List[float],
    residual_scale: float = 0.15,
    min_size: float = 0.02,
) -> List[float]:
    """Upstream rollout.py:49 — final = sanitize(coarse + residual_scale * tanh(raw)).

    tanh bounds every residual to (-residual_scale, residual_scale), so a raw
    action of any magnitude can move each box edge by at most residual_scale.
    """
    if len(coarse_box) != 4 or len(raw_action) != 4:
        raise ValueError("coarse_box and raw_action must both have 4 entries")
    residual = [residual_scale * math.tanh(v) for v in raw_action]
    return sanitize_normalized_box(
        [c + r for c, r in zip(coarse_box, residual)], min_size
    )


def apply_action_transform(
    coarse_box: List[float],
    raw_action: List[float],
    scale: float = 0.2,
) -> List[float]:
    """
    Legacy prototype transform (NOT upstream-aligned; kept for reference only).
    Upstream uses 0.15 * tanh(raw) residuals plus a center/size sanitize with
    min box side 0.02 — see apply_nv_cot_action_transform above.
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

    # --- upstream-port checks ---
    # 1. text box (0-1000 units) -> coarse normalized box
    coarse = text_box_to_coarse([200, 200, 600, 600])
    assert all(abs(a - b) < 1e-9 for a, b in zip(coarse, [0.2, 0.2, 0.6, 0.6])), coarse

    # 2. residual is bounded by residual_scale regardless of raw magnitude
    final = apply_nv_cot_action_transform([0.4, 0.4, 0.6, 0.6], [100.0, -100.0, 100.0, -100.0])
    assert abs((final[0] - 0.4) - 0.15) < 1e-6, final  # tanh(100) ~ 1
    assert abs((final[1] - 0.4) + 0.15) < 1e-6, final
    assert abs((final[2] - 0.6) - 0.15) < 1e-6, final
    assert abs((final[3] - 0.6) + 0.15) < 1e-6, final

    # 3. reversed coarse box is sanitized (center/size), not hard-sorted
    rev = apply_nv_cot_action_transform([0.6, 0.6, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0])
    assert all(abs(a - b) < 1e-9 for a, b in zip(rev, [0.2, 0.2, 0.6, 0.6])), rev

    # 4. out-of-frame box is shifted back into [0, 1] with min side preserved
    oob = apply_nv_cot_action_transform([0.9, 0.9, 1.3, 1.3], [0.0, 0.0, 0.0, 0.0])
    assert all(abs(a - b) < 1e-9 for a, b in zip(oob, [0.6, 0.6, 1.0, 1.0])), oob

    # 5. tiny/degenerate box is expanded to min_size
    tiny = apply_nv_cot_action_transform([0.5, 0.5, 0.5001, 0.5001], [0.0, 0.0, 0.0, 0.0])
    assert tiny[2] - tiny[0] >= 0.02 - 1e-9, tiny
    assert tiny[3] - tiny[1] >= 0.02 - 1e-9, tiny

    # 6. zero residual is identity on an already-valid box
    ident = apply_nv_cot_action_transform([0.2, 0.2, 0.6, 0.6], [0.0, 0.0, 0.0, 0.0])
    assert all(abs(a - b) < 1e-9 for a, b in zip(ident, [0.2, 0.2, 0.6, 0.6])), ident

    # 7. shape validation
    try:
        apply_nv_cot_action_transform([0.1, 0.1, 0.5], [0.0, 0.0, 0.0, 0.0])
        raise AssertionError("malformed box should have raised ValueError")
    except ValueError:
        pass

    # --- legacy prototype checks (reference only) ---
    legacy_final = apply_action_transform([0.2, 0.2, 0.6, 0.6], [0.1, -0.1, 0.2, 0.1], scale=0.2)
    assert is_valid_box(legacy_final), "Box should be valid!"
    assert abs(compute_box_iou(legacy_final, legacy_final) - 1.0) < 1e-5
    iou = compute_box_iou(legacy_final, [0.22, 0.18, 0.65, 0.62])
    assert iou > 0.8, f"Expected high IoU, got {iou}"

    print(f"Upstream transform checks passed! text[200,200,600,600] -> coarse {coarse} -> final {final}")
    print(f"Legacy prototype check passed! IoU: {iou:.3f}")
    print("All unit tests in eval_regions.py passed successfully!")
