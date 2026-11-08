"""Training-free ViCrop baseline on Qwen3-VL-2B (S0 / BASE2B) under our exact 0.25MP overview + 1MP crop budget.

Computes the question-conditioned visual-token saliency map from a single prefill pass on the 0.25MP overview,
selects the highest-activation sliding window on the 0..1000 grid, crops that region at 1MP from the original
image, and answers in <final>...</final>.
"""
import json
import os
import sys
from pathlib import Path
import torch
from transformers import StoppingCriteriaList

WS = Path("/data/ibu-bi-ai-share-disk/zk/project/coord-opd/v1_workspace")
BASE = WS / "probe10_20261028"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(WS))

tag, ckpt, bench = sys.argv[1:4]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.environ["LOWRES_OV_PIXELS"] = "262144"

import v1_lowres_pipeline as P  # noqa: E402
from v1_core import (  # noqa: E402
    FINAL_CLOSE, FINAL_OPEN, FOLLOWUP_PROMPT, SYSTEM_PROMPT, OrderedBoxHead,
    StopAtBoxPrefixOrTerminal, encode_chat, execute_crop_1000, format_canonical_box_suffix,
    CANONICAL_BOX_PREFIX, load_student_model, to_device,
)


@torch.no_grad()
def _vicrop_select_box(vlm, processor, messages, orig_image, device):
    inp = encode_chat(processor, messages, [orig_image], max_pixels=262144)
    d = to_device(inp, device)
    ids = d["input_ids"][0]
    thw = d.get("image_grid_thw")
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        out = vlm.model(**d, use_cache=False, output_hidden_states=True, return_dict=True)
    # Use middle-to-late layer hidden-state cosine alignment between question tokens and visual tokens
    hs = out.hidden_states[len(out.hidden_states) * 2 // 3][0].float()
    # Locate image pad tokens (id 151655 in Qwen-VL family, or non-ASCII vision span)
    vis_id = getattr(vlm.config, "image_token_id", 151655)
    vis_pos = (ids == vis_id).nonzero(as_tuple=False).squeeze(-1)
    if vis_pos.numel() < 4 or thw is None:
        return [150, 150, 850, 850]
    gh = int(thw[0, 1].item()) // 2
    gw = int(thw[0, 2].item()) // 2
    if gh * gw != int(vis_pos.numel()) or gh < 2 or gw < 2:
        return [150, 150, 850, 850]
    q_start = int(vis_pos[-1].item()) + 1
    q_vec = F.normalize(hs[q_start:].mean(dim=0, keepdim=True), dim=-1)
    v_vecs = F.normalize(hs[vis_pos], dim=-1)
    sal = torch.matmul(v_vecs, q_vec.t()).squeeze(-1).view(gh, gw)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
    # Sliding window of ~0.45 x 0.45 of page (standard ViCrop zoom ratio)
    wh = max(1, int(round(gh * 0.45)))
    ww = max(1, int(round(gw * 0.45)))
    best_val, best_r, best_c = -1e9, 0, 0
    for r in range(0, gh - wh + 1):
        for c in range(0, gw - ww + 1):
            val = float(sal[r:r + wh, c:c + ww].mean().item())
            if val > best_val:
                best_val, best_r, best_c = val, r, c
    x1 = int(round(1000.0 * best_c / gw))
    y1 = int(round(1000.0 * best_r / gh))
    x2 = int(round(1000.0 * (best_c + ww) / gw))
    y2 = int(round(1000.0 * (best_r + wh) / gh))
    return [max(0, min(990, x1)), max(0, min(990, y1)), max(10, min(1000, x2)), max(10, min(1000, y2))]


@torch.no_grad()
def _vicrop_rollout(
    vlm, head, processor, orig_image, user_text, execution_mode, decode_mode, device,
    orig_max_pixels=1048576, **kwargs
):
    vlm.eval()
    messages = [
        dict(role="system", content=SYSTEM_PROMPT),
        dict(role="user", content=[dict(type="image"), dict(type="text", text=user_text)]),
    ]
    q_box = _vicrop_select_box(vlm, processor, messages, orig_image, device)
    exec_info = execute_crop_1000(orig_image, q_box)
    tool_str = CANONICAL_BOX_PREFIX + format_canonical_box_suffix(exec_info["quantized_box"])
    messages2 = list(messages) + [
        dict(role="assistant", content=tool_str),
        dict(role="user", content=[dict(type="image"), dict(type="text", text=FOLLOWUP_PROMPT)]),
    ]
    images2 = [orig_image, exec_info["crop"]]
    pref_inputs = encode_chat(processor, messages2, images2, suffix=FINAL_OPEN, max_pixels=orig_max_pixels)
    prompt_len = int(pref_inputs["input_ids"].shape[-1])
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        gen = vlm.generate(
            **to_device(pref_inputs, device),
            do_sample=False,
            max_new_tokens=128,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([
                StopAtBoxPrefixOrTerminal(processor.tokenizer, prompt_len, stop_on_box_prefix=False)
            ]),
            pad_token_id=processor.tokenizer.pad_token_id,
        )[0, prompt_len:].cpu()
    text = processor.tokenizer.decode(gen.tolist(), skip_special_tokens=True)
    ans = text.split(FINAL_CLOSE, 1)[0].strip() if FINAL_CLOSE in text else text.strip()
    return dict(
        turns=[],
        actions=[dict(turn_idx=0, action_idx=0, quantized_box=exec_info["quantized_box"],
                      executed_pixel_box=exec_info["executed_pixel_box"], executed_norm_box=exec_info["executed_norm_box"])],
        executed_boxes_1000=[exec_info["quantized_box"]],
        crop_count=1,
        final_answer=ans,
        termination_reason="completed_final",
        format_errors=0,
        protocol_success=bool(ans),
        total_new_tokens=int(gen.numel()),
    )


P.rollout_trajectory = _vicrop_rollout

dev = "cuda:0"
if ckpt == "BASE2B":
    vlm, head, proc = load_student_model(P.P_2B, device=dev, checkpoint_dir=None, train_lora=False, train_head=False, seed=42)
else:
    vlm, head, proc = load_student_model(P.P_2B, device=dev, checkpoint_dir=str(WS / "checkpoints" / ckpt),
                                         train_lora=False, train_head=False, seed=42)

s = P.evaluate_model_on_benchmark(vlm, head, proc, bench, execution_mode="text", run_id=tag, step=0,
                                  device=dev, limit=limit)
row = dict(s, tag=tag, ckpt=ckpt, ov=262144, vicrop=True)
with (BASE / "lowres/eval_rows.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("EVAL_DONE", tag, ckpt, bench, "acc", s["answer_accuracy"], "n", s["n_rows"], "crops", s["mean_crops"],
      "protofail", s["protocol_failure_rate"], flush=True)
