"""
v1_core.py: Core specification module for Coord-OPD v1.

Implements:
1. Official Qwen3-VL-8B-Instruct (Teacher) and Qwen3-VL-2B-Instruct (Student) loading
   with language-model-only LoRA (r=16, alpha=32, dropout=0) and strict freeze checks.
2. OrderedBoxHead: LayerNorm(d) -> Linear(d, 512) -> GELU -> Linear(512, 4) with
   ordered sigmoid parameterization (delta=0.001).
3. Unified crop tool protocol (<tool_call>{"name":"crop","arguments":{"image_id":0,"bbox":[x1,y1,x2,y2]}}</tool_call>
   and <final>...</final>) with shared 0..1000 integer grid quantization (floor(1000*b+0.5))
   and 2-pixel minimum expansion on original RGB images.
4. Chunked (64-token) activation-recomputed full-vocabulary reverse KL (L_KL_all, L_KL_common, W),
   independent teacher-forcing full-box CE branch (L_boxCE), and FP32 L1+GIoU geometry loss (L_geom).
5. Official evaluation metrics for V*, TreeBench, HR-Bench 4K, and HR-Bench 8K (including cyclic accuracy
   and multi-target executed-crop localization metrics at tau=0.3 and 0.5).
"""

import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file, save_file

SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. You can inspect details in the original image "
    "by calling the crop tool up to 3 times:\n"
    '<tool_call>{"name":"crop","arguments":{"image_id":0,"bbox":[x1,y1,x2,y2]}}</tool_call>\n'
    "where x1,y1,x2,y2 are integers in 0..1000 relative to the original image (image_id 0). "
    "When you have enough visual evidence, provide your final answer inside:\n"
    "<final>YOUR_ANSWER</final>"
)

FOLLOWUP_PROMPT = (
    "Above is the cropped region from image_id 0. Call the crop tool again if needed (up to 3 total crops), "
    "or provide your final answer in <final>YOUR_ANSWER</final>."
)

FORCE_FINAL_PROMPT = (
    "You have used your crop budget. Now provide your final answer immediately inside <final>YOUR_ANSWER</final>."
)

FORMAT_FEEDBACK_PROMPT = (
    'Invalid format. Output either a valid crop call <tool_call>{"name":"crop","arguments":{"image_id":0,"bbox":[x1,y1,x2,y2]}}</tool_call> '
    "or your final answer <final>YOUR_ANSWER</final>."
)

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"
CANONICAL_BOX_PREFIX = '<tool_call>{"name":"crop","arguments":{"image_id":0,"bbox":['
CANONICAL_BOX_SUFFIX = "]}}</tool_call>"


def format_canonical_box_suffix(q: List[int]) -> str:
    return f"{int(q[0])},{int(q[1])},{int(q[2])},{int(q[3])}" + CANONICAL_BOX_SUFFIX


assert format_canonical_box_suffix([1, 2, 3, 4]) == "1,2,3,4]}}</tool_call>"
BOX_PREFIX_RE = re.compile(r'<tool_call>\s*\{\s*"name"\s*:\s*"crop"\s*,\s*"arguments"\s*:\s*\{\s*"image_id"\s*:\s*0\s*,\s*"bbox"\s*:\s*\[')
FINAL_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL)
OPTION_LETTER_RE = re.compile(r"^\s*\(?([A-K])\)?(?:\s|[.):,-]|$)", re.I)

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class CropProtocolError(ValueError):
    """Raised when a model turn or crop box violates the unified v1 protocol."""


class OrderedBoxHead(nn.Module):
    """2-layer MLP reading last hidden state at '..."bbox":[' and outputting ordered xyxy in [0, 1]^4."""

    def __init__(self, hidden_size: int, mlp_width: int = 512, delta: float = 0.001):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.mlp_width = int(mlp_width)
        self.delta = float(delta)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.mlp_width),
            nn.GELU(),
            nn.Linear(self.mlp_width, 4),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-1.0, -1.0, 0.0, 0.0], dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = hidden.float()
        raw = self.net(h)
        a, b, c, d = raw.unbind(dim=-1)
        d_min = self.delta
        x1 = (1.0 - d_min) * torch.sigmoid(a)
        y1 = (1.0 - d_min) * torch.sigmoid(b)
        x2 = x1 + d_min + (1.0 - x1 - d_min) * torch.sigmoid(c)
        y2 = y1 + d_min + (1.0 - y1 - d_min) * torch.sigmoid(d)
        return torch.stack([x1, y1, x2, y2], dim=-1)


def get_text_hidden_size(vlm) -> int:
    return int(getattr(vlm.config, "hidden_size", None) or vlm.config.text_config.hidden_size)


def load_teacher_model(model_path: str, device: str = "cuda:0"):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=True)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    vlm.requires_grad_(False)
    vlm.eval()
    return vlm, processor


def load_student_model(
    model_path: str,
    device: str = "cuda:0",
    checkpoint_dir: Optional[str] = None,
    train_lora: bool = True,
    train_head: bool = True,
    seed: int = 42,
):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=True)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    vlm.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    inject_adapter_in_model(lora_cfg, vlm.model.language_model)

    # Strictly enforce freeze rules: only lora_ parameters inside vlm.model.language_model may train
    for name, p in vlm.named_parameters():
        if "model.language_model." in name and "lora_" in name:
            p.requires_grad_(bool(train_lora))
        else:
            p.requires_grad_(False)

    d = get_text_hidden_size(vlm)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = OrderedBoxHead(d, mlp_width=512, delta=0.001).to(device)

    if checkpoint_dir is not None:
        ckpt = Path(checkpoint_dir)
        lora_file = ckpt / "lora_weights.safetensors"
        head_file = ckpt / "coord_head.safetensors"
        if lora_file.exists():
            lora_sd = load_file(str(lora_file), device=device)
            missing, unexpected = vlm.load_state_dict(lora_sd, strict=False)
            assert not unexpected, f"Unexpected keys in lora_weights: {unexpected}"
        if head_file.exists():
            head.load_state_dict(load_file(str(head_file), device=device), strict=True)

    head.requires_grad_(bool(train_head))
    if train_lora or train_head:
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.eval()
    head.eval()
    return vlm, head, processor


def verify_student_trainable_scope(vlm, head) -> Dict[str, Any]:
    vlm_trainable = [n for n, p in vlm.named_parameters() if p.requires_grad]
    head_trainable = [n for n, p in head.named_parameters() if p.requires_grad]
    for n in vlm_trainable:
        assert n.startswith("model.language_model."), f"Illegal trainable module outside language_model: {n}"
        assert "lora_" in n, f"Non-LoRA parameter marked trainable: {n}"
        assert "visual" not in n and "embed_tokens" not in n and "lm_head" not in n
    vlm_count = sum(p.numel() for n, p in vlm.named_parameters() if p.requires_grad)
    head_count = sum(p.numel() for n, p in head.named_parameters() if p.requires_grad)
    return {
        "vlm_trainable_tensors": len(vlm_trainable),
        "vlm_trainable_params": int(vlm_count),
        "head_trainable_tensors": len(head_trainable),
        "head_trainable_params": int(head_count),
        "sample_vlm_trainable_names": vlm_trainable[:6],
        "head_trainable_names": head_trainable,
    }


def save_student_checkpoint(vlm, head, out_dir: Path, metadata: Optional[Dict[str, Any]] = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_sd = {n: p.detach().cpu().contiguous() for n, p in vlm.named_parameters() if "lora_" in n}
    head_sd = {n: p.detach().cpu().contiguous() for n, p in head.state_dict().items()}
    save_file(lora_sd, str(out_dir / "lora_weights.safetensors"))
    save_file(head_sd, str(out_dir / "coord_head.safetensors"))
    if metadata is not None:
        (out_dir / "checkpoint_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def load_record_image(rec: Dict[str, Any]) -> Image.Image:
    if rec.get("image_path"):
        raw = Path(rec["image_path"]).read_bytes()
    else:
        with open(rec["image_pack"], "rb") as f:
            f.seek(int(rec["image_offset"]))
            raw = f.read(int(rec["image_size"]))
    with Image.open(io.BytesIO(raw)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def quantize_float_box_to_1000(b_float: List[float]) -> List[int]:
    if len(b_float) != 4 or not all(math.isfinite(float(v)) for v in b_float):
        raise CropProtocolError("Non-finite or non-4D float box")
    q = [int(math.floor(1000.0 * float(v) + 0.5)) for v in b_float]
    q[0] = max(0, min(999, q[0]))
    q[1] = max(0, min(999, q[1]))
    q[2] = max(q[0] + 1, min(1000, q[2]))
    q[3] = max(q[1] + 1, min(1000, q[3]))
    return q


def execute_crop_1000(orig_image: Image.Image, box_1000: List[Any], raw_box: Optional[List[float]] = None) -> Dict[str, Any]:
    """Unified 0..1000 integer grid crop executor on original RGB image."""
    if not isinstance(box_1000, (list, tuple)) or len(box_1000) != 4:
        raise CropProtocolError("bbox must be a 4-element list")
    try:
        vals = [float(v) for v in box_1000]
    except Exception as exc:
        raise CropProtocolError(f"Non-numeric bbox: {box_1000}") from exc
    if not all(math.isfinite(v) for v in vals):
        raise CropProtocolError("Non-finite bbox coordinates")
    # Reject out-of-range or unordered coordinates without silently fixing them
    if vals[0] < 0 or vals[1] < 0 or vals[2] > 1000 or vals[3] > 1000:
        raise CropProtocolError(f"bbox out of 0..1000 range: {vals}")
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        raise CropProtocolError(f"Unordered or degenerate bbox: {vals}")

    q = [int(math.floor(v + 0.5)) for v in vals]
    q[0] = max(0, min(999, q[0]))
    q[1] = max(0, min(999, q[1]))
    q[2] = max(q[0] + 1, min(1000, q[2]))
    q[3] = max(q[1] + 1, min(1000, q[3]))

    w, h = orig_image.size
    px1 = max(0, min(w, int(math.floor(q[0] * w / 1000.0))))
    py1 = max(0, min(h, int(math.floor(q[1] * h / 1000.0))))
    px2 = max(0, min(w, int(math.ceil(q[2] * w / 1000.0))))
    py2 = max(0, min(h, int(math.ceil(q[3] * h / 1000.0))))

    if px2 - px1 < 2:
        cx = (px1 + px2) // 2
        px1 = max(0, min(w - 2, cx - 1))
        px2 = min(w, px1 + 2)
    if py2 - py1 < 2:
        cy = (py1 + py2) // 2
        py1 = max(0, min(h - 2, cy - 1))
        py2 = min(h, py1 + 2)

    crop = orig_image.crop((px1, py1, px2, py2))
    crop_sha = hashlib.sha256(f"{crop.mode}:{crop.width}x{crop.height}:".encode() + crop.tobytes()).hexdigest()
    exec_norm = [px1 / w, py1 / h, px2 / w, py2 / h]
    return dict(
        raw_box=list(raw_box) if raw_box is not None else [v / 1000.0 for v in vals],
        quantized_box=q,
        executed_pixel_box=[px1, py1, px2, py2],
        executed_norm_box=exec_norm,
        crop=crop,
        crop_sha256=crop_sha,
        crop_wh=[crop.width, crop.height],
    )


def box_area_tensor(b: torch.Tensor) -> torch.Tensor:
    return (b[..., 2] - b[..., 0]).clamp(min=0.0) * (b[..., 3] - b[..., 1]).clamp(min=0.0)


def compute_giou_tensor(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Exact GIoU in FP32 for normalized xyxy boxes b1, b2 of shape [..., 4]."""
    b1 = b1.float()
    b2 = b2.float()
    ix1 = torch.maximum(b1[..., 0], b2[..., 0])
    iy1 = torch.maximum(b1[..., 1], b2[..., 1])
    ix2 = torch.minimum(b1[..., 2], b2[..., 2])
    iy2 = torch.minimum(b1[..., 3], b2[..., 3])
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

    a1 = box_area_tensor(b1)
    a2 = box_area_tensor(b2)
    union = (a1 + a2 - inter).clamp(min=1e-9)
    iou = inter / union

    cx1 = torch.minimum(b1[..., 0], b2[..., 0])
    cy1 = torch.minimum(b1[..., 1], b2[..., 1])
    cx2 = torch.maximum(b1[..., 2], b2[..., 2])
    cy2 = torch.maximum(b1[..., 3], b2[..., 3])
    c_area = ((cx2 - cx1).clamp(min=0.0) * (cy2 - cy1).clamp(min=0.0)).clamp(min=1e-9)
    return iou - (c_area - union) / c_area


def compute_geom_loss(b_student_float: torch.Tensor, b_teacher_exec: torch.Tensor) -> Dict[str, torch.Tensor]:
    """L_geom = mean(|bS_float - stopgrad(bT_exec)|) + (1 - GIoU(bS_float, stopgrad(bT_exec))) in FP32."""
    bs = b_student_float.float()
    bt = b_teacher_exec.detach().float().to(bs.device)
    l1 = (bs - bt).abs().mean(dim=-1)
    giou = compute_giou_tensor(bs, bt)
    loss_giou = 1.0 - giou
    total = (l1 + loss_giou).mean()
    return dict(loss=total, l1=l1.mean().detach(), giou=giou.mean().detach())


def encode_chat(
    processor,
    messages: List[Dict[str, Any]],
    images: List[Image.Image],
    suffix: str = "",
    max_pixels: int = 1048576,
    min_pixels: int = 4096,
):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + suffix
    return processor(
        text=[prompt],
        images=images if images else None,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        return_tensors="pt",
    )


def to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def forward_last_hidden(vlm, inputs: Dict[str, Any], device: str) -> torch.Tensor:
    data = to_device(inputs, device)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        out = vlm.model(**data, use_cache=False, return_dict=True)
    return out.last_hidden_state


def _kl_chunk_fn(s_h: torch.Tensor, t_h: torch.Tensor, w: torch.Tensor, s_lm_head: nn.Module, t_lm_head: nn.Module) -> torch.Tensor:
    """Weighted sum of token-level reverse KL D_KL(p_S || p_T) on one chunk of hidden states."""
    s_logits = s_lm_head(s_h).float()
    with torch.no_grad():
        t_logits = t_lm_head(t_h).float()
        t_logits = torch.where(torch.isfinite(t_logits), t_logits, torch.zeros_like(t_logits))
        log_pt = F.log_softmax(t_logits, dim=-1)
    log_ps = F.log_softmax(s_logits, dim=-1)
    ps = log_ps.exp()
    token_kl = (ps * (log_ps - log_pt)).sum(dim=-1)
    return (token_kl * w).sum()


def compute_turn_kl_loss(
    student_vlm,
    teacher_vlm,
    prefix_inputs: Dict[str, Any],
    response_ids: List[int],
    token_weights: List[float],
    device: str,
    chunk_size: int = 64,
    use_recompute: bool = True,
) -> Tuple[torch.Tensor, float]:
    """Full-vocab reverse KL over one turn's assistant tokens, chunked with activation recomputation."""
    L = len(response_ids)
    assert L == len(token_weights) and L > 0
    w_tensor = torch.tensor(token_weights, dtype=torch.float32, device=device)
    w_sum = float(w_tensor.sum().item())
    if w_sum <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0

    data = to_device(prefix_inputs, device)
    prefix_len = data["input_ids"].shape[-1]
    resp = torch.tensor([response_ids], dtype=data["input_ids"].dtype, device=device)
    full_data = dict(data)
    full_data["input_ids"] = torch.cat([data["input_ids"], resp], dim=-1)
    full_data["attention_mask"] = torch.ones_like(full_data["input_ids"])

    with torch.no_grad():
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            t_out = teacher_vlm.model(**full_data, use_cache=False, return_dict=True)
            t_hidden = t_out.last_hidden_state[0, prefix_len - 1 : prefix_len - 1 + L].detach()

    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        s_out = student_vlm.model(**full_data, use_cache=False, return_dict=True)
        s_hidden = s_out.last_hidden_state[0, prefix_len - 1 : prefix_len - 1 + L]

    total_kl = None
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        sh_chunk = s_hidden[start:end]
        th_chunk = t_hidden[start:end]
        w_chunk = w_tensor[start:end]
        if float(w_chunk.sum().item()) == 0.0:
            continue
        if use_recompute and sh_chunk.requires_grad:
            term = grad_checkpoint(
                _kl_chunk_fn, sh_chunk, th_chunk, w_chunk, student_vlm.lm_head, teacher_vlm.lm_head, use_reentrant=False
            )
        else:
            term = _kl_chunk_fn(sh_chunk, th_chunk, w_chunk, student_vlm.lm_head, teacher_vlm.lm_head)
        total_kl = term if total_kl is None else total_kl + term

    if total_kl is None:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0
    return total_kl, w_sum


def compute_box_ce_branch(
    student_vlm,
    pre_box_inputs: Dict[str, Any],
    target_suffix_ids: List[int],
    device: str,
) -> torch.Tensor:
    """L_boxCE: Independent teacher-forcing CE branch on pre-box state h for target 'x1,y1,x2,y2]}</tool_call>'."""
    L = len(target_suffix_ids)
    assert L > 0
    data = to_device(pre_box_inputs, device)
    prefix_len = data["input_ids"].shape[-1]
    target = torch.tensor([target_suffix_ids], dtype=data["input_ids"].dtype, device=device)
    full_data = dict(data)
    full_data["input_ids"] = torch.cat([data["input_ids"], target], dim=-1)
    full_data["attention_mask"] = torch.ones_like(full_data["input_ids"])

    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        s_out = student_vlm.model(**full_data, use_cache=False, return_dict=True)
        s_hidden = s_out.last_hidden_state[0, prefix_len - 1 : prefix_len - 1 + L]
    logits = student_vlm.lm_head(s_hidden).float()
    return F.cross_entropy(logits, target[0], reduction="mean")


def box_iou_float(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def box_coverage_float(crop_box: List[float], gt_box: List[float]) -> float:
    ix1, iy1 = max(crop_box[0], gt_box[0]), max(crop_box[1], gt_box[1])
    ix2, iy2 = min(crop_box[2], gt_box[2]), min(crop_box[3], gt_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ga = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
    return inter / ga if ga > 0.0 else 0.0


def compute_multi_target_localization(
    gt_boxes_norm: List[List[float]],
    executed_boxes_norm: List[List[float]],
    answer_correct: bool,
) -> Dict[str, Any]:
    """Section 3 localization metrics over executed crops C and ground-truth targets G."""
    if not gt_boxes_norm:
        return dict(has_valid_gt=False, crop_count=len(executed_boxes_norm))

    def eval_crops(crops: List[List[float]]) -> Dict[str, Any]:
        if not crops:
            return dict(
                target_miou=0.0,
                target_recall_03=0.0,
                target_recall_05=0.0,
                all_targets_hit_03=0.0,
                all_targets_hit_05=0.0,
                joint_03=0.0,
                joint_05=0.0,
                crop_precision_iou=0.0,
                mean_coverage=0.0,
            )
        per_g_best_iou = []
        per_g_best_cov = []
        for g in gt_boxes_norm:
            per_g_best_iou.append(max(box_iou_float(g, c) for c in crops))
            per_g_best_cov.append(max(box_coverage_float(c, g) for c in crops))
        per_c_best_iou = [max(box_iou_float(c, g) for g in gt_boxes_norm) for c in crops]
        hit03 = [float(u >= 0.3) for u in per_g_best_iou]
        hit05 = [float(u >= 0.5) for u in per_g_best_iou]
        all03 = float(all(u >= 0.3 for u in per_g_best_iou))
        all05 = float(all(u >= 0.5 for u in per_g_best_iou))
        return dict(
            target_miou=sum(per_g_best_iou) / len(per_g_best_iou),
            target_recall_03=sum(hit03) / len(hit03),
            target_recall_05=sum(hit05) / len(hit05),
            all_targets_hit_03=all03,
            all_targets_hit_05=all05,
            joint_03=float(bool(answer_correct) and bool(all03 == 1.0)),
            joint_05=float(bool(answer_correct) and bool(all05 == 1.0)),
            crop_precision_iou=sum(per_c_best_iou) / len(per_c_best_iou),
            mean_coverage=sum(per_g_best_cov) / len(per_g_best_cov),
        )

    first_stats = eval_crops(executed_boxes_norm[:1])
    all_stats = eval_crops(executed_boxes_norm)
    return dict(
        has_valid_gt=True,
        num_targets=len(gt_boxes_norm),
        crop_count=len(executed_boxes_norm),
        first_crop=first_stats,
        all_crops=all_stats,
    )


def score_prediction(pred_final: Optional[str], gt_answer: str, options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Deterministic exact-match scoring without LLM judge fallback."""
    if pred_final is None or not str(pred_final).strip():
        return dict(correct=False, pred_norm=None, reason="missing_final")
    p = str(pred_final).strip()
    g = str(gt_answer).strip()

    if options and g.upper() in options:
        # Multiple-choice question
        m = OPTION_LETTER_RE.match(p)
        letter = m.group(1).upper() if m else None
        if letter is None:
            # Check if exact option text matches uniquely
            matches = [k for k, v in options.items() if v.strip().lower() == p.lower()]
            if len(matches) == 1:
                letter = matches[0].upper()
        if letter is None:
            return dict(correct=False, pred_norm=p[:40], reason="invalid_mc_option")
        return dict(correct=(letter == g.upper()), pred_norm=letter, reason="ok")

    # Open-ended short QA normalized match
    p_low = re.sub(r"\s+", " ", p.lower()).strip(" .!,")
    g_low = re.sub(r"\s+", " ", g.lower()).strip(" .!,")
    if g_low.startswith("yes,") or g_low == "yes":
        ok = p_low == "yes" or p_low.startswith("yes,") or p_low.startswith("yes ")
    elif g_low.startswith("no,") or g_low == "no":
        ok = p_low == "no" or p_low.startswith("no,") or p_low.startswith("no ")
    else:
        ok = (
            (p_low == g_low)
            or (len(g_low) >= 2 and g_low in p_low)
            or (len(p_low) >= 2 and bool(re.search(rf"\b{re.escape(p_low)}\b", g_low)))
        )
    return dict(correct=ok, pred_norm=p[:80], reason="ok")


def format_user_question(question: str, options: Optional[Dict[str, str]] = None) -> str:
    q = str(question).strip()
    if options:
        opt_lines = "\n".join(f"{k}. {v}" for k, v in options.items() if str(v).strip())
        return f"Question: {q}\nOptions:\n{opt_lines}"
    return f"Question: {q}"


class StopAtBoxPrefixOrTerminal(StoppingCriteria):
    """Stop generation when '</final>', '</tool_call>', or the box prefix '...\"bbox\":[' is completed."""

    def __init__(self, tokenizer, prompt_len: int, stop_on_box_prefix: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.stop_on_box_prefix = bool(stop_on_box_prefix)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.prompt_len :].tolist()
        if not gen_ids:
            return False
        tail_ids = gen_ids[-48:]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
        if FINAL_CLOSE in tail_text or TOOL_CLOSE in tail_text:
            return True
        if self.stop_on_box_prefix and '"bbox":[' in tail_text.replace(" ", ""):
            if tail_text.rstrip().endswith("["):
                return True
        return False


class StopAtToolClose(StoppingCriteria):
    """Stop greedy coordinate completion as soon as '</tool_call>' is emitted."""

    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.prompt_len :].tolist()
        if not gen_ids:
            return False
        tail_text = self.tokenizer.decode(gen_ids[-24:], skip_special_tokens=False)
        return TOOL_CLOSE in tail_text


def parse_box_from_tool_text(tool_text: str) -> List[float]:
    s = tool_text.find(TOOL_OPEN)
    e = tool_text.find(TOOL_CLOSE, s + len(TOOL_OPEN)) if s >= 0 else -1
    if s < 0 or e < 0:
        raise CropProtocolError("Missing <tool_call>...</tool_call> tags")
    body = tool_text[s + len(TOOL_OPEN) : e].strip()
    if body.endswith("]}") and not body.endswith("]}}") and body.count("{") == body.count("}") + 1:
        body = body + "}"
    try:
        obj = json.loads(body)
    except Exception as exc:
        raise CropProtocolError(f"Invalid tool_call JSON: {body[:100]}") from exc
    if not isinstance(obj, dict) or obj.get("name") != "crop":
        raise CropProtocolError(f"Unsupported tool name in {body[:80]}")
    args = obj.get("arguments")
    if not isinstance(args, dict) or int(args.get("image_id", -1)) != 0:
        raise CropProtocolError(f"Invalid image_id (must be 0): {body[:80]}")
    bbox = args.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise CropProtocolError(f"Invalid bbox list: {bbox}")
    return [float(v) for v in bbox]


@torch.no_grad()
def query_teacher_box(
    teacher_vlm,
    processor,
    pre_box_inputs: Dict[str, Any],
    orig_image: Image.Image,
    device: str,
    max_new_tokens: int = 96,
) -> Dict[str, Any]:
    """Query frozen teacher at pre-box state h ('...\"bbox\":[') to greedily complete the full box."""
    data = to_device(pre_box_inputs, device)
    prefix_len = int(data["input_ids"].shape[-1])
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        gen = teacher_vlm.generate(
            **data,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([StopAtToolClose(processor.tokenizer, prefix_len)]),
            pad_token_id=processor.tokenizer.pad_token_id,
        )[0, prefix_len:].cpu()
    suffix_text = processor.tokenizer.decode(gen.tolist(), skip_special_tokens=False)
    suffix_text = suffix_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    full_tool_text = CANONICAL_BOX_PREFIX + suffix_text
    try:
        raw_vals = parse_box_from_tool_text(full_tool_text)
        exec_info = execute_crop_1000(orig_image, raw_vals)
        q = exec_info["quantized_box"]
        canonical_suffix = format_canonical_box_suffix(q)
        target_ids = processor.tokenizer.encode(canonical_suffix, add_special_tokens=False)
        return dict(
            valid=True,
            error=None,
            quantized_box=q,
            executed_pixel_box=exec_info["executed_pixel_box"],
            executed_norm_box=exec_info["executed_norm_box"],
            canonical_suffix=canonical_suffix,
            target_suffix_ids=target_ids,
            teacher_generated_tokens=len(gen),
            raw_suffix_text=suffix_text[:120],
        )
    except CropProtocolError as exc:
        return dict(
            valid=False,
            error=str(exc),
            quantized_box=None,
            executed_pixel_box=None,
            executed_norm_box=None,
            canonical_suffix=None,
            target_suffix_ids=[],
            teacher_generated_tokens=len(gen),
            raw_suffix_text=suffix_text[:120],
        )


@torch.no_grad()
def predict_head_box_at_prefix(
    vlm,
    head: OrderedBoxHead,
    pre_box_inputs: Dict[str, Any],
    orig_image: Image.Image,
    device: str,
) -> Dict[str, Any]:
    """Run forward pass on pre_box_inputs ending at '...\"bbox\":[' and predict full box with OrderedBoxHead."""
    hidden = forward_last_hidden(vlm, pre_box_inputs, device)
    last_h = hidden[0, -1]
    b_float = head(last_h).detach().cpu().tolist()
    q = quantize_float_box_to_1000(b_float)
    exec_info = execute_crop_1000(orig_image, q, raw_box=b_float)
    return dict(
        b_float=b_float,
        quantized_box=q,
        executed_pixel_box=exec_info["executed_pixel_box"],
        executed_norm_box=exec_info["executed_norm_box"],
        crop=exec_info["crop"],
        crop_sha256=exec_info["crop_sha256"],
        crop_wh=exec_info["crop_wh"],
    )


@torch.no_grad()
def rollout_trajectory(
    vlm,
    head: OrderedBoxHead,
    processor,
    orig_image: Image.Image,
    user_text: str,
    execution_mode: str,
    decode_mode: str,
    device: str,
    orig_max_pixels: int = 1048576,
    crop_max_pixels: int = 1048576,
    max_crops: int = 3,
    max_turns: int = 5,
    max_new_tokens_per_turn: int = 512,
    max_new_tokens_total: int = 2048,
    max_context_tokens: int = 8192,
    max_format_feedback: int = 1,
    override_first_box_1000: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Unified rollout engine for 'text' and 'continuous_head' modes."""
    assert execution_mode in ("text", "continuous_head")
    assert decode_mode in ("train", "eval")
    training = decode_mode == "train"
    vlm.eval()
    head.eval()

    messages = [
        dict(role="system", content=SYSTEM_PROMPT),
        dict(role="user", content=[dict(type="image"), dict(type="text", text=user_text)]),
    ]
    images = [orig_image]
    turns = []
    actions = []
    final_answer = None
    termination_reason = "max_turns_reached"
    format_errors = 0
    total_new_tokens = 0

    for turn_idx in range(max_turns):
        prefix_inputs = encode_chat(processor, messages, images, max_pixels=orig_max_pixels)
        prompt_len = int(prefix_inputs["input_ids"].shape[-1])
        rem_tokens = min(max_new_tokens_per_turn, max_new_tokens_total - total_new_tokens)
        if rem_tokens <= 0 or prompt_len + rem_tokens > max_context_tokens:
            termination_reason = "context_or_token_budget_exceeded"
            break

        # Stage 1: generate free text up to '...\"bbox\":[' or '</final>' or '</tool_call>'
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            gen1 = vlm.generate(
                **to_device(prefix_inputs, device),
                do_sample=training,
                temperature=1.0,
                top_p=1.0 if training else 1.0,
                max_new_tokens=rem_tokens,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([
                    StopAtBoxPrefixOrTerminal(processor.tokenizer, prompt_len, stop_on_box_prefix=True)
                ]),
                pad_token_id=processor.tokenizer.pad_token_id,
            )[0, prompt_len:].cpu()

        gen1_ids = gen1.tolist()
        text1 = processor.tokenizer.decode(gen1_ids, skip_special_tokens=False)
        text1_clean = text1.replace("<|im_end|>", "").replace("<|endoftext|>", "")

        # Check if we stopped at a valid box prefix '...\"bbox\":['
        m_box = BOX_PREFIX_RE.search(text1_clean)
        if m_box and FINAL_CLOSE not in text1_clean[: m_box.start()]:
            # Truncate cleanly right after '"bbox":['
            pre_box_text = text1_clean[: m_box.end()]
            pre_box_inputs = encode_chat(processor, messages, images, suffix=pre_box_text, max_pixels=orig_max_pixels)
            pre_box_len = int(pre_box_inputs["input_ids"].shape[-1])
            pre_box_resp_ids = pre_box_inputs["input_ids"][0, prompt_len:].tolist()

            if len(actions) >= max_crops:
                # Exceeded crop budget
                turns.append(dict(
                    turn_idx=turn_idx,
                    messages=[dict(m) for m in messages],
                    images=list(images),
                    prefix_len=prompt_len,
                    response_ids=pre_box_resp_ids,
                    non_coord_mask=[1.0] * len(pre_box_resp_ids),
                    all_lm_mask=[1.0] * len(pre_box_resp_ids),
                    coord_weighted_mask=[1.0] * len(pre_box_resp_ids),
                    text=pre_box_text,
                    kind="exceeded_crop_budget",
                ))
                total_new_tokens += len(pre_box_resp_ids)
                termination_reason = "exceeded_crop_budget"
                break

            if execution_mode == "continuous_head":
                head_out = predict_head_box_at_prefix(vlm, head, pre_box_inputs, orig_image, device)
                q = head_out["quantized_box"]
                b_float = head_out["b_float"]
                if override_first_box_1000 is not None and len(actions) == 0:
                    exec_info = execute_crop_1000(orig_image, override_first_box_1000, raw_box=b_float)
                    q = exec_info["quantized_box"]
                else:
                    exec_info = head_out
                forced_suffix = format_canonical_box_suffix(q)
                forced_ids = processor.tokenizer.encode(forced_suffix, add_special_tokens=False)
                full_resp_ids = pre_box_resp_ids + forced_ids
                # Forced coordinate suffix has weight 0 in ALL text KL masks!
                non_coord_mask = [1.0] * len(pre_box_resp_ids) + [0.0] * len(forced_ids)
                all_lm_mask = list(non_coord_mask)
                coord_weighted_mask = list(non_coord_mask)
                full_turn_text = pre_box_text + forced_suffix
                total_new_tokens += len(pre_box_resp_ids)
            else:
                # Text mode: greedily decode the coordinate payload from pre_box_inputs
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    gen2 = vlm.generate(
                        **to_device(pre_box_inputs, device),
                        do_sample=False,
                        max_new_tokens=48,
                        use_cache=True,
                        stopping_criteria=StoppingCriteriaList([StopAtToolClose(processor.tokenizer, pre_box_len)]),
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )[0, pre_box_len:].cpu()
                coord_ids = gen2.tolist()
                coord_text = processor.tokenizer.decode(coord_ids, skip_special_tokens=False)
                coord_text_clean = coord_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                close_pos = coord_text_clean.find(TOOL_CLOSE)
                if close_pos >= 0:
                    coord_text_clean = coord_text_clean[: close_pos + len(TOOL_CLOSE)]
                    coord_ids = processor.tokenizer.encode(coord_text_clean, add_special_tokens=False)
                full_turn_text = pre_box_text + coord_text_clean
                full_resp_ids = pre_box_resp_ids + coord_ids
                non_coord_mask = [1.0] * len(pre_box_resp_ids) + [0.0] * len(coord_ids)
                all_lm_mask = [1.0] * len(pre_box_resp_ids) + [1.0] * len(coord_ids)
                coord_weighted_mask = [1.0] * len(pre_box_resp_ids) + [4.0] * len(coord_ids)
                total_new_tokens += len(full_resp_ids)

                try:
                    raw_vals = parse_box_from_tool_text(full_turn_text[m_box.start() :])
                    if override_first_box_1000 is not None and len(actions) == 0:
                        exec_info = execute_crop_1000(orig_image, override_first_box_1000)
                    else:
                        exec_info = execute_crop_1000(orig_image, raw_vals)
                    b_float = [v / 1000.0 for v in exec_info["quantized_box"]]
                except CropProtocolError as exc:
                    turns.append(dict(
                        turn_idx=turn_idx,
                        messages=[dict(m) for m in messages],
                        images=list(images),
                        prefix_len=prompt_len,
                        response_ids=full_resp_ids,
                        non_coord_mask=non_coord_mask,
                        all_lm_mask=all_lm_mask,
                        coord_weighted_mask=coord_weighted_mask,
                        text=full_turn_text,
                        kind="invalid_crop_call",
                        error=str(exc),
                    ))
                    format_errors += 1
                    if format_errors > max_format_feedback:
                        termination_reason = f"protocol_error:{exc}"
                        break
                    messages.extend([
                        dict(role="assistant", content=full_turn_text),
                        dict(role="user", content=[dict(type="text", text=FORMAT_FEEDBACK_PROMPT)]),
                    ])
                    continue

            turns.append(dict(
                turn_idx=turn_idx,
                messages=[dict(m) for m in messages],
                images=list(images),
                prefix_len=prompt_len,
                response_ids=full_resp_ids,
                non_coord_mask=non_coord_mask,
                all_lm_mask=all_lm_mask,
                coord_weighted_mask=coord_weighted_mask,
                text=full_turn_text,
                kind="crop",
            ))
            actions.append(dict(
                turn_idx=turn_idx,
                action_idx=len(actions),
                pre_box_messages=[dict(m) for m in messages],
                pre_box_images=list(images),
                pre_box_suffix=pre_box_text,
                b_float=b_float,
                quantized_box=exec_info["quantized_box"],
                executed_pixel_box=exec_info["executed_pixel_box"],
                executed_norm_box=exec_info["executed_norm_box"],
                crop_sha256=exec_info["crop_sha256"],
                crop_wh=exec_info["crop_wh"],
                crop=exec_info["crop"],
            ))
            images.append(exec_info["crop"])
            next_prompt = FORCE_FINAL_PROMPT if len(actions) >= max_crops else FOLLOWUP_PROMPT
            messages.extend([
                dict(role="assistant", content=full_turn_text),
                dict(role="user", content=[dict(type="image"), dict(type="text", text=next_prompt)]),
            ])
            continue

        # Not a box prefix: check if it contains <final>...</final>
        total_new_tokens += len(gen1_ids)
        m_final = FINAL_RE.search(text1_clean)
        if m_final and m_final.group(1).strip():
            final_answer = m_final.group(1).strip()
            end_pos = m_final.end()
            clean_final_text = text1_clean[:end_pos]
            resp_ids = processor.tokenizer.encode(clean_final_text, add_special_tokens=False)
            turns.append(dict(
                turn_idx=turn_idx,
                messages=[dict(m) for m in messages],
                images=list(images),
                prefix_len=prompt_len,
                response_ids=resp_ids,
                non_coord_mask=[1.0] * len(resp_ids),
                all_lm_mask=[1.0] * len(resp_ids),
                coord_weighted_mask=[1.0] * len(resp_ids),
                text=clean_final_text,
                kind="final",
            ))
            termination_reason = "completed_final"
            break

        # Invalid format turn
        turns.append(dict(
            turn_idx=turn_idx,
            messages=[dict(m) for m in messages],
            images=list(images),
            prefix_len=prompt_len,
            response_ids=gen1_ids,
            non_coord_mask=[1.0] * len(gen1_ids),
            all_lm_mask=[1.0] * len(gen1_ids),
            coord_weighted_mask=[1.0] * len(gen1_ids),
            text=text1_clean,
            kind="format_error",
        ))
        format_errors += 1
        if format_errors > max_format_feedback:
            termination_reason = "protocol_error:missing_tool_or_final"
            break
        messages.extend([
            dict(role="assistant", content=text1_clean),
            dict(role="user", content=[dict(type="text", text=FORMAT_FEEDBACK_PROMPT)]),
        ])

    return dict(
        turns=turns,
        actions=actions,
        final_answer=final_answer,
        termination_reason=termination_reason,
        format_errors=format_errors,
        total_new_tokens=total_new_tokens,
        crop_count=len(actions),
        protocol_success=(final_answer is not None and termination_reason == "completed_final"),
    )


def summarize_benchmark_predictions(bench_rows: List[Dict[str, Any]], pred_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute official benchmark summary (including HR-Bench cyclic accuracy and multi-target localization)."""
    pred_by_id = {r["id"]: r for r in pred_records}
    n_rows = len(bench_rows)
    assert len(pred_by_id) == n_rows, f"Expected {n_rows} predictions, got {len(pred_by_id)}"

    n_correct = 0
    n_no_crop = 0
    n_proto_fail = 0
    total_crops = 0
    groups = {}
    image_groups = set()

    first_miou, all_miou = [], []
    rec03, rec05 = [], []
    allhit03, allhit05 = [], []
    joint03, joint05 = [], []
    prec_iou = []

    for row in bench_rows:
        p = pred_by_id[row["id"]]
        c = bool(p["correct"])
        n_correct += int(c)
        n_no_crop += int(p["crop_count"] == 0)
        n_proto_fail += int(not p.get("protocol_success", True))
        total_crops += int(p["crop_count"])
        gid = row.get("group_id", row["id"])
        groups.setdefault(gid, []).append(c)
        image_groups.add(row.get("rgb_sha256", gid))

        loc = p.get("localization", {})
        if loc.get("has_valid_gt"):
            fc = loc["first_crop"]
            ac = loc["all_crops"]
            first_miou.append(fc["target_miou"])
            all_miou.append(ac["target_miou"])
            rec03.append(ac["target_recall_03"])
            rec05.append(ac["target_recall_05"])
            allhit03.append(ac["all_targets_hit_03"])
            allhit05.append(ac["all_targets_hit_05"])
            joint03.append(ac["joint_03"])
            joint05.append(ac["joint_05"])
            prec_iou.append(ac["crop_precision_iou"])

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    bench_name = (bench_rows[0].get("benchmark") or bench_rows[0].get("split", "")) if bench_rows else ""
    is_cyclic = bench_name.startswith("hrbench")
    cyclic_acc = round(sum(1 for g in groups.values() if all(g)) / len(groups), 4) if is_cyclic and groups else None

    return dict(
        benchmark=bench_name,
        n_rows=n_rows,
        n_base_questions=len(groups),
        n_image_groups=len(image_groups),
        n_correct=n_correct,
        answer_accuracy=round(n_correct / n_rows, 4) if n_rows else 0.0,
        cyclic_accuracy=cyclic_acc,
        target_miou_first_crop=avg(first_miou),
        target_miou_all_crops=avg(all_miou),
        target_recall_iou03=avg(rec03),
        target_recall_iou05=avg(rec05),
        all_targets_hit_iou03=avg(allhit03),
        all_targets_hit_iou05=avg(allhit05),
        joint_iou03=avg(joint03),
        joint_iou05=avg(joint05),
        crop_precision_iou=avg(prec_iou),
        n_with_valid_gt=len(all_miou),
        n_no_crop=n_no_crop,
        mean_crops=round(total_crops / n_rows, 4) if n_rows else 0.0,
        protocol_failure_rate=round(n_proto_fail / n_rows, 4) if n_rows else 0.0,
    )

