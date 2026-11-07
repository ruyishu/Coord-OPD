"""
v1_overnight_pipeline.py: End-to-End Automated Overnight Master Pipeline for Coord-OPD v1.

Coordinates Machine A (--role a) and Machine B (--role b) via shared JuiceFS:
1. Waits for Step 2B (splits) and Step 3-5 (numerical & memory checks) to finish.
2. Role A trains the shared cold-start checkpoint S0 on warmup_10k (joint text CE + head L_geom),
   verifies protocol success rate on dev_fast_200, builds state_probe.jsonl, and runs
   teacher_box_replacement.jsonl diagnostics. Signals S0/READY.
3. Role A and Role B then execute the formal OPD runs and benchmark evaluations in parallel:
   - Role A runs: B_s42 -> A_s42 -> W_s42 -> B_s17 -> B_s2026
   - Role B runs: T (Teacher eval) & S0_T / S0_H eval -> C_s42 -> D_s42 -> H0_s42 -> C_s17 -> C_s2026
4. Continuously writes per-step training logs to coord_opd_results/train/<run_id>.jsonl,
   per-example benchmark predictions to coord_opd_results/eval/per_example/,
   and updates coord_opd_results/eval/summary.csv and paired_comparisons.csv.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.nn import functional as F

ROOT = Path(".")
WS = ROOT / "v1_workspace"
sys.path.insert(0, str(WS))

import lowres_patch  # noqa: F401  (must come before any 'from v1_core import encode_chat')
from tk_core import (QKL_STATS, build_qkl_candidate_boxes, build_qkl_feasible_sets, compute_turn_kl_loss_2v,
                     compute_turn_kl_loss_hs, compute_turn_kl_loss_qkl, ef_candidate_loss, make_tk_hook,
                     rollout_trajectory_tk, tk_branch_kl)
import geo_feas as GF

GEO_STATS = dict(n_pos=0, n_empty=0, sum_fsize=0, n_ok=0, n_infeasible=0, n_format=0, n_unknown=0, n_turns=0)
_GEO_CACHE = {}
from coord_exp import flush_diag, install_attention_capture, m2_term, mask_without_coord_digits, p_term
import geo_exp as GEO
from geo_exp import box_terms, coverage_cost_fn, gt_cost_fn
from v1_core import (
    CANONICAL_BOX_PREFIX,
    FOLLOWUP_PROMPT,
    SYSTEM_PROMPT,
    OrderedBoxHead,
    box_iou_float,
    compute_box_ce_branch,
    compute_geom_loss,
    compute_multi_target_localization,
    compute_turn_kl_loss,
    encode_chat,
    execute_crop_1000,
    format_canonical_box_suffix,
    format_user_question,
    forward_last_hidden,
    load_record_image,
    load_student_model,
    load_teacher_model,
    predict_head_box_at_prefix,
    quantize_float_box_to_1000,
    query_teacher_box,
    rollout_trajectory,
    save_student_checkpoint,
    score_prediction,
    summarize_benchmark_predictions,
    to_device,
)

_OPD_LR = float(os.environ.get("OPD_LR", "1e-5"))
DATA_DIR = WS / "data"
BENCH_DIR = DATA_DIR / "benchmarks"
SPLIT_DIR = DATA_DIR / "splits"
CKPT_ROOT = WS / "checkpoints"
RES_DIR = WS / "coord_opd_results"
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

H = Path("./.cache/huggingface/hub")
P_8B = str(H / "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
P_2B = str(H / "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203")


def log_msg(role: str, msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][Role-{role.upper()}] {msg}", flush=True)


def wait_for_file(path: Path, role: str, desc: str, poll_sec: int = 10):
    while not path.exists():
        log_msg(role, f"Waiting for {desc} ({path})...")
        time.sleep(poll_sec)
    log_msg(role, f"Found {desc}: {path}")


def load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit > 0 else rows


def train_cold_start_s0(role: str, device: str = "cuda:0") -> Path:
    s0_dir = CKPT_ROOT / "S0"
    ready_flag = s0_dir / "READY"
    if ready_flag.exists():
        log_msg(role, "S0 checkpoint already READY, skipping S0 training.")
        return s0_dir

    log_msg(role, "=== STEP 6: Training Shared Cold-Start S0 on warmup_10k ===")
    warmup_rows = [r for r in load_jsonl(SPLIT_DIR / "warmup_10k.jsonl") if r.get("quantized_box_1000")]
    random.Random(42).shuffle(warmup_rows)
    student_vlm, student_head, student_proc = load_student_model(
        P_2B, device=device, train_lora=True, train_head=True, seed=42
    )
    student_vlm.eval()

    head_params = list(student_head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": 3e-4}],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    eff_batch = 16
    target_updates = 128
    warmup_steps = max(1, int(0.05 * target_updates))

    def lr_scale(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        prog = float(step_idx - warmup_steps) / float(max(1, target_updates - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    s0_log_path = RES_DIR / "train/S0_warmup.jsonl"
    cursor = 0
    t_start = time.monotonic()

    with s0_log_path.open("w", encoding="utf-8") as f_log:
        for upd in range(1, target_updates + 1):
            t0 = time.monotonic()
            scale = lr_scale(upd - 1)
            optimizer.param_groups[0]["lr"] = 3e-4 * scale
            optimizer.zero_grad(set_to_none=True)
            student_head.train()

            geo_losses, l1_losses, giou_losses = [], [], []
            valid_trajs = 0

            for _ in range(eff_batch):
                row = warmup_rows[cursor % len(warmup_rows)]
                cursor += 1
                try:
                    orig_im = load_record_image(row)
                except Exception:
                    continue
                u_text = format_user_question(row["question"], row.get("options"))
                q = row["quantized_box_1000"]
                exec_info = execute_crop_1000(orig_im, q)
                b_gt_exec = torch.tensor(exec_info["executed_norm_box"], dtype=torch.float32, device=device)

                msgs_t0 = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]},
                ]
                pre_in_t0 = encode_chat(student_proc, msgs_t0, [orig_im], suffix=CANONICAL_BOX_PREFIX, max_pixels=524288)
                with torch.no_grad():
                    h_last = forward_last_hidden(student_vlm, pre_in_t0, device)[0, -1].detach()
                b_pred = student_head(h_last)
                g_out = compute_geom_loss(b_pred, b_gt_exec)
                geo_t0 = g_out["loss"]
                (geo_t0 / eff_batch).backward()
                geo_losses.append(float(geo_t0.item()))
                l1_losses.append(float(g_out["l1"].item()))
                giou_losses.append(1.0 - float(g_out["giou"].item()))
                valid_trajs += 1

            g_norm = float(torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0).item())
            optimizer.step()
            dt = time.monotonic() - t0
            rec_log = dict(
                stage="S0_warmup",
                update=upd,
                valid_trajectories=valid_trajs,
                loss_ce=0.0,
                loss_geom=round(sum(geo_losses) / max(1, len(geo_losses)), 4),
                loss_l1=round(sum(l1_losses) / max(1, len(l1_losses)), 4),
                loss_giou=round(sum(giou_losses) / max(1, len(giou_losses)), 4),
                grad_norm=round(g_norm, 4),
                lr_head=optimizer.param_groups[0]["lr"],
                step_sec=round(dt, 2),
                elapsed_sec=round(time.monotonic() - t_start, 1),
            )
            f_log.write(json.dumps(rec_log) + "\n")
            f_log.flush()
            if upd % 16 == 0 or upd == 1 or upd == target_updates:
                log_msg(role, f"S0 update {upd}/{target_updates}: geo={rec_log['loss_geom']:.4f} (l1={rec_log['loss_l1']:.4f}, giou={rec_log['loss_giou']:.4f}) grad={g_norm:.3f} ({dt:.1f}s)")

    save_student_checkpoint(student_vlm, student_head, s0_dir, metadata={"updates": target_updates, "samples": cursor})
    ready_flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log_msg(role, f"S0 checkpoint saved and marked READY at {s0_dir}")
    del optimizer, student_vlm, student_head
    torch.cuda.empty_cache()
    return s0_dir


@torch.no_grad()
def evaluate_model_on_benchmark(
    vlm,
    head: OrderedBoxHead,
    processor,
    bench_name: str,
    execution_mode: str,
    run_id: str,
    step: int,
    device: str = "cuda:0",
    orig_max_pixels: int = 1048576,
    limit: int = 0,
) -> Dict[str, Any]:
    _lr_p = WS / "probe10_20261028/lowres/data" / f"{bench_name}.jsonl"
    bench_path = _lr_p if _lr_p.exists() else (BENCH_DIR / f"{bench_name}.jsonl" if (BENCH_DIR / f"{bench_name}.jsonl").exists() else SPLIT_DIR / f"{bench_name}.jsonl")
    rows = load_jsonl(bench_path, limit=limit)
    budget_tag = "1mp" if orig_max_pixels == 1048576 else f"{orig_max_pixels}px"
    if os.environ.get("LOWRES_OV_PIXELS", "0") not in ("", "0"):
        budget_tag += "_ov" + os.environ["LOWRES_OV_PIXELS"]
    out_jsonl = RES_DIR / f"eval/per_example/{run_id}_step{step}__{bench_name}__{budget_tag}.jsonl"

    existing = {}
    if out_jsonl.exists():
        for l in out_jsonl.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    o = json.loads(l)
                    existing[o["id"]] = o
                except Exception:
                    pass

    pred_records = []
    t0 = time.monotonic()
    with out_jsonl.open("a", encoding="utf-8") as f_out:
        for idx, row in enumerate(rows):
            if row["id"] in existing:
                pred_records.append(existing[row["id"]])
                continue
            orig_im = load_record_image(row)
            u_text = format_user_question(row["question"], row.get("options"))
            traj = rollout_trajectory(
                vlm, head, processor, orig_im, u_text,
                execution_mode=execution_mode,
                decode_mode="eval",
                device=device,
                orig_max_pixels=orig_max_pixels,
                crop_max_pixels=1048576,
            )
            scored = score_prediction(traj["final_answer"], row["answer"], row.get("options"))
            if not scored["correct"]:
                for _pa in ((row.get("label_audit") or {}).get("possible_answers") or []):
                    if _pa and score_prediction(traj["final_answer"], str(_pa), row.get("options"))["correct"]:
                        scored = dict(scored, correct=True)
                        break
            exec_boxes_norm = [a["executed_norm_box"] for a in traj["actions"]]
            loc = compute_multi_target_localization(row.get("gt_boxes_norm", []), exec_boxes_norm, scored["correct"])
            rec = dict(
                id=row["id"],
                group_id=row.get("group_id", row["id"]),
                benchmark=bench_name,
                category=row.get("category", ""),
                final_answer=traj["final_answer"],
                pred_norm=scored["pred_norm"],
                gt_answer=row["answer"],
                correct=bool(scored["correct"]),
                protocol_success=bool(traj["protocol_success"]),
                termination_reason=traj["termination_reason"],
                crop_count=traj["crop_count"],
                executed_boxes_1000=[a["quantized_box"] for a in traj["actions"]],
                executed_boxes_norm=exec_boxes_norm,
                crop_sha256s=[a["crop_sha256"] for a in traj["actions"]],
                localization=loc,
            )
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            pred_records.append(rec)

    summary = summarize_benchmark_predictions(rows, pred_records)
    summary.update(
        benchmark=bench_name,
        run_id=run_id,
        step=step,
        execution_mode=execution_mode,
        visual_budget=budget_tag,
        eval_seconds=round(time.monotonic() - t0, 1),
    )
    return summary


def run_step7_8_diagnostics(role: str, s0_dir: Path, device: str = "cuda:0"):
    diag_probe_path = RES_DIR / "diagnostics/state_probe.jsonl"
    diag_repl_path = RES_DIR / "diagnostics/teacher_box_replacement.jsonl"
    if diag_probe_path.exists() and diag_repl_path.exists():
        log_msg(role, "Step 7 & 8 diagnostics already exist, skipping.")
        return

    log_msg(role, "=== STEP 7 & 8: Running state_probe and Teacher Box Replacement Diagnostics on dev_fast_200 ===")
    teacher_vlm, teacher_proc = load_teacher_model(P_8B, device=device)
    student_vlm, student_head, student_proc = load_student_model(
        P_2B, device=device, checkpoint_dir=str(s0_dir), train_lora=False, train_head=False, seed=42
    )
    dev_rows = load_jsonl(SPLIT_DIR / "dev_fast_200.jsonl", limit=100)

    probe_records = []
    repl_records = []

    for idx, row in enumerate(dev_rows):
        orig_im = load_record_image(row)
        u_text = format_user_question(row["question"], row.get("options"))
        # Run S0-T to get natural first crop state
        traj_s = rollout_trajectory(
            student_vlm, student_head, student_proc, orig_im, u_text,
            execution_mode="text", decode_mode="eval", device=device,
        )
        scored_s = score_prediction(traj_s["final_answer"], row["answer"], row.get("options"))
        if not traj_s["actions"]:
            continue

        act0 = traj_s["actions"][0]
        pre_in = encode_chat(
            student_proc, act0["pre_box_messages"], act0["pre_box_images"],
            suffix=act0["pre_box_suffix"], max_pixels=1048576,
        )
        t_box_info = query_teacher_box(teacher_vlm, teacher_proc, pre_in, orig_im, device)
        head_info = predict_head_box_at_prefix(student_vlm, student_head, pre_in, orig_im, device)

        gt_boxes = row.get("gt_boxes_norm", [])
        s_norm = act0["executed_norm_box"]
        h_norm = head_info["executed_norm_box"]
        t_norm = t_box_info["executed_norm_box"] if t_box_info["valid"] else None

        probe_rec = dict(
            id=row["id"],
            teacher_valid=t_box_info["valid"],
            student_text_box_1000=act0["quantized_box"],
            student_head_box_1000=head_info["quantized_box"],
            teacher_box_1000=t_box_info["quantized_box"],
            iou_student_text_vs_teacher=box_iou_float(s_norm, t_norm) if t_norm else None,
            iou_student_head_vs_teacher=box_iou_float(h_norm, t_norm) if t_norm else None,
            iou_student_text_vs_gt=max((box_iou_float(s_norm, g) for g in gt_boxes), default=None),
            iou_student_head_vs_gt=max((box_iou_float(h_norm, g) for g in gt_boxes), default=None),
            iou_teacher_vs_gt=max((box_iou_float(t_norm, g) for g in gt_boxes), default=None) if t_norm else None,
        )
        probe_records.append(probe_rec)

        if t_box_info["valid"] and len(repl_records) < 60:
            # Enlarge student box by 1.2x around center
            q_s = act0["quantized_box"]
            cx, cy = 0.5 * (q_s[0] + q_s[2]), 0.5 * (q_s[1] + q_s[3])
            bw, bh = (q_s[2] - q_s[0]) * 1.2, (q_s[3] - q_s[1]) * 1.2
            q_enl = [
                max(0, min(999, int(round(cx - 0.5 * bw)))),
                max(0, min(999, int(round(cy - 0.5 * bh)))),
                max(1, min(1000, int(round(cx + 0.5 * bw)))),
                max(1, min(1000, int(round(cy + 0.5 * bh)))),
            ]
            if q_enl[2] <= q_enl[0]:
                q_enl[2] = min(1000, q_enl[0] + 2)
            if q_enl[3] <= q_enl[1]:
                q_enl[3] = min(1000, q_enl[1] + 2)

            traj_tbox = rollout_trajectory(
                student_vlm, student_head, student_proc, orig_im, u_text,
                execution_mode="text", decode_mode="eval", device=device,
                override_first_box_1000=t_box_info["quantized_box"],
            )
            traj_enl = rollout_trajectory(
                student_vlm, student_head, student_proc, orig_im, u_text,
                execution_mode="text", decode_mode="eval", device=device,
                override_first_box_1000=q_enl,
            )
            sc_tbox = score_prediction(traj_tbox["final_answer"], row["answer"], row.get("options"))
            sc_enl = score_prediction(traj_enl["final_answer"], row["answer"], row.get("options"))
            repl_records.append(dict(
                id=row["id"],
                student_box_correct=bool(scored_s["correct"]),
                teacher_box_correct=bool(sc_tbox["correct"]),
                enlarged_box_correct=bool(sc_enl["correct"]),
                student_box_1000=q_s,
                teacher_box_1000=t_box_info["quantized_box"],
                enlarged_box_1000=q_enl,
            ))

    diag_probe_path.write_text("\n".join(json.dumps(r) for r in probe_records) + "\n", encoding="utf-8")
    diag_repl_path.write_text("\n".join(json.dumps(r) for r in repl_records) + "\n", encoding="utf-8")
    log_msg(role, f"Diagnostics saved: state_probe={len(probe_records)}, replacement={len(repl_records)}")
    del teacher_vlm, student_vlm, student_head
    torch.cuda.empty_cache()


def run_opd_branch(
    role: str,
    run_id: str,
    method: str,
    seed: int,
    s0_dir: Path,
    total_updates: int = 128,
    eff_batch: int = 8,
    lambda_weight: float = 1.0,
    device: str = "cuda:0",
    offline_cache_dir: Optional[Path] = None,
    evaluate: bool = True,
) -> List[Dict[str, Any]]:
    """Run one formal OPD training branch (A, B, C, D, W, or H0) and evaluate its checkpoints."""
    assert method in ("A", "WH", "HS", "HS5", "AT", "ATG", "EF", "QK", "QK1", "QKG", "B", "C", "D", "W", "H0", "M2", "P", "V", "VC", "G1", "G2", "TK1", "TK2", "TK3", "TKN", "TKV2")
    exec_mode = "continuous_head" if method in ("C", "D", "H0") else "text"
    train_head = method in ("C", "D")

    run_ckpt_dir = CKPT_ROOT / run_id
    final_ckpt_dir = run_ckpt_dir / f"step_{total_updates}"
    if (final_ckpt_dir / "READY").exists():
        log_msg(role, f"{run_id} already completed at {final_ckpt_dir}, proceeding to evaluation.")
    else:
        log_msg(role, f"=== Starting OPD Run {run_id} (method={method}, seed={seed}, exec={exec_mode}, updates={total_updates}) ===")
        torch.manual_seed(seed)
        random.seed(seed)

        teacher_vlm, teacher_proc = load_teacher_model(P_8B, device=device)
        student_vlm, student_head, student_proc = load_student_model(
            P_2B, device=device, checkpoint_dir=str(s0_dir), train_lora=True, train_head=train_head, seed=seed
        )
        # If D (offline S0-H trajectories), also keep a frozen S0-H model or generate S0-H trajectories once
        s0_vlm, s0_head = None, None
        if method == "D":
            s0_vlm, s0_head, _ = load_student_model(
                P_2B, device=device, checkpoint_dir=str(s0_dir), train_lora=False, train_head=False, seed=seed
            )

        ref_vlm, attn_cap = None, None
        diag_path = RES_DIR / f"train/{run_id}.coord_diag.jsonl"
        if method == "M2":
            ref_vlm, _ref_head, _ = load_student_model(P_2B, device=device, checkpoint_dir=str(s0_dir), train_lora=False, train_head=False, seed=seed)
            ref_vlm.eval()
            for _p in ref_vlm.parameters():
                _p.requires_grad_(False)
        if method == "P":
            attn_cap = install_attention_capture(student_vlm, student_proc)
        lora_params = [p for p in student_vlm.parameters() if p.requires_grad]
        head_params = [p for p in student_head.parameters() if p.requires_grad]
        param_groups = [{"params": lora_params, "lr": _OPD_LR}]
        if head_params:
            param_groups.append({"params": head_params, "lr": 1e-3})
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)

        opd_rows = load_jsonl(Path(os.environ["GEO_POOL"]) if os.environ.get("GEO_POOL") else SPLIT_DIR / "opd_pool_10k.jsonl")
        rng = random.Random(seed)
        order = list(range(len(opd_rows)))
        rng.shuffle(order)

        warmup_steps = max(1, int(0.03 * total_updates))

        def lr_scale(step_idx: int) -> float:
            if step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            prog = float(step_idx - warmup_steps) / float(max(1, total_updates - warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        train_log_path = RES_DIR / f"train/{run_id}.jsonl"
        ckpt_steps = {int(round(total_updates * frac)) for frac in (0.125, 0.25, 0.5, 1.0)}
        cursor = 0
        t_run_start = time.monotonic()

        with train_log_path.open("w", encoding="utf-8") as f_log:
            for upd in range(1, total_updates + 1):
                t_step = time.monotonic()
                scale = lr_scale(upd - 1)
                optimizer.param_groups[0]["lr"] = _OPD_LR * scale
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = 1e-3 * scale

                optimizer.zero_grad(set_to_none=True)
                if method in ("QK", "QK1", "QKG"):
                    QKL_STATS.update(n_pos=0, n_empty=0, sum_fsize=0, n_candidates=0, n_turns=0)
                    GEO_STATS.update(n_pos=0, n_empty=0, sum_fsize=0, n_ok=0, n_infeasible=0, n_format=0, n_unknown=0, n_turns=0)
                batch_kl_losses = []
                batch_act_losses = []
                batch_crops = 0
                batch_proto_ok = 0
                batch_tokens = 0
                teacher_queries = 0
                teacher_valid_boxes = 0
                lr_ious = []
                lr_resets = []
                batch_tk_box = []

                for _ in range(eff_batch):
                    row = opd_rows[order[cursor % len(order)]]
                    cursor += 1
                    try:
                        orig_im = load_record_image(row)
                    except Exception:
                        continue
                    u_text = format_user_question(row["question"], row.get("options"))

                    # 1. Rollout trajectory (from current student, or frozen S0-H for D)
                    roll_vlm = s0_vlm if method == "D" else student_vlm
                    roll_head = s0_head if method == "D" else student_head
                    if method in ("TK1", "TK2", "TK3", "TKN", "TKV2", "AT", "ATG", "EF", "QK", "QK1", "QKG"):
                        traj = rollout_trajectory_tk(
                            roll_vlm, roll_head, student_proc, orig_im, u_text,
                            execution_mode=exec_mode, decode_mode="train", device=device,
                            teacher_hook=make_tk_hook("QK" if method in ("QK1", "QKG") else method, teacher_vlm, teacher_proc, orig_im, device),
                        )
                    else:
                        traj = rollout_trajectory(
                            roll_vlm, roll_head, student_proc, orig_im, u_text,
                            execution_mode=exec_mode, decode_mode="train", device=device,
                        )
                    batch_crops += traj["crop_count"]
                    batch_proto_ok += int(traj["protocol_success"])

                    student_vlm.train()
                    if train_head:
                        student_head.train()

                    # 2. Text OPD loss across all turns in trajectory (normalized within trajectory)
                    traj_kl_sum = None
                    traj_weight_sum = 0.0
                    for turn in traj["turns"]:
                        if not turn["response_ids"]:
                            continue
                        if method in ("A", "WH", "HS", "HS5", "AT", "ATG", "EF", "QK", "QK1", "QKG", "V", "VC", "G1", "G2"):
                            w_mask = turn["all_lm_mask"]
                        elif method in ("TK1", "TK2", "TK3", "TKN", "TKV2"):
                            _ta = (next((a for a in traj["actions"] if a["turn_idx"] == turn["turn_idx"]), None)
                                   if turn.get("kind") == "crop" else None)
                            _ok = _ta is not None and _ta.get("tk") is not None and _ta["tk"]["valid"]
                            w_mask = turn["non_coord_mask"] if _ok else turn["all_lm_mask"]
                        elif method == "W":
                            w_mask = turn["coord_weighted_mask"]
                        elif method in ("M2", "P"):
                            w_mask = mask_without_coord_digits(turn, student_proc.tokenizer)
                        else:
                            w_mask = turn["non_coord_mask"]
                        if sum(w_mask) <= 0.0:
                            continue
                        pref_in = encode_chat(student_proc, turn["messages"], turn["images"], max_pixels=1048576)
                        if method == "WH":
                            t_pref_in = lowres_patch._ORIG_ENCODE_CHAT(teacher_proc, turn["messages"], list(turn["images"]), max_pixels=1048576)
                            kl_term, w_sum = compute_turn_kl_loss_2v(
                                student_vlm, teacher_vlm, pref_in, t_pref_in, turn["response_ids"], w_mask, device, chunk_size=64, use_recompute=True
                            )
                        elif method in ("QK", "QK1", "QKG"):
                            _ta_q = (next((a for a in traj["actions"] if a["turn_idx"] == turn["turn_idx"]), None)
                                     if turn.get("kind") == "crop" else None)
                            _tk_q = _ta_q.get("tk") if _ta_q else None
                            t_pref_in = lowres_patch._ORIG_ENCODE_CHAT(teacher_proc, turn["messages"], list(turn["images"]), max_pixels=1048576)
                            _qw = 0.5 if method == "QK" else 1.0
                            if _tk_q and _tk_q.get("valid"):
                                coord_ids = [rid for rid, m in zip(turn["response_ids"], turn["non_coord_mask"]) if m == 0.0]
                                if method == "QKG":
                                    # Exact geometric feasible sets (geo_feas); any failure falls back to ordinary KL at that turn.
                                    try:
                                        if "gram" not in _GEO_CACHE:
                                            _GEO_CACHE["gram"] = GF.GeoGrammar(student_proc.tokenizer, 1000)
                                        _res = _GEO_CACHE["gram"].analyze(coord_ids, _tk_q["t_info"]["quantized_box"])
                                        fsets = [sorted(a) for a, _ in _res]
                                        GEO_STATS["n_turns"] += 1
                                        GEO_STATS["n_pos"] += len(fsets)
                                        GEO_STATS["n_empty"] += sum(not f for f in fsets)
                                        GEO_STATS["sum_fsize"] += sum(len(f) for f in fsets)
                                        GEO_STATS["n_ok"] += sum(s == "ok" for _, s in _res)
                                        GEO_STATS["n_infeasible"] += sum(s == "infeasible" for _, s in _res)
                                        GEO_STATS["n_format"] += sum(s == "format" for _, s in _res)
                                    except Exception as _geo_exc:
                                        print("GEO_UNKNOWN_FALLBACK", repr(_geo_exc)[:160], flush=True)
                                        GEO_STATS["n_unknown"] += 1
                                        fsets = [[] for _ in coord_ids]
                                else:
                                    candidates = build_qkl_candidate_boxes(_tk_q["t_info"]["quantized_box"], _ta_q["quantized_box"])
                                    fsets = build_qkl_feasible_sets(student_proc.tokenizer, _ta_q["pre_box_suffix"], coord_ids, candidates)
                                f_iter = iter(fsets)
                                feasible_by_pos = [None if m > 0.0 else next(f_iter) for m in turn["non_coord_mask"]]
                                kl_term, w_sum = compute_turn_kl_loss_qkl(
                                    student_vlm, teacher_vlm, pref_in, t_pref_in, turn["response_ids"], turn["non_coord_mask"],
                                    feasible_by_pos, _qw, device, chunk_size=64, use_recompute=True
                                )
                            else:
                                kl_term, w_sum = compute_turn_kl_loss_hs(
                                    student_vlm, teacher_vlm, pref_in, t_pref_in, turn["response_ids"], turn["non_coord_mask"],
                                    _qw, device, chunk_size=64, use_recompute=True
                                )
                        elif method in ("HS", "HS5", "AT", "ATG", "EF"):
                            _ta_h = (next((a for a in traj["actions"] if a["turn_idx"] == turn["turn_idx"]), None)
                                     if turn.get("kind") == "crop" else None)
                            _tk_h = _ta_h.get("tk") if _ta_h else None
                            if method == "HS":
                                _cw = 1.0
                            elif method == "HS5":
                                _cw = 0.5
                            elif method in ("AT", "ATG"):
                                _cw = 0.0 if (_tk_h and _tk_h.get("in_at")) else 0.5
                            else:
                                _cw = 0.0 if (_tk_h and _tk_h.get("ef_accept")) else 0.5
                            if _cw > 0.0 and any(m == 0.0 for m in turn["non_coord_mask"]):
                                t_pref_in = lowres_patch._ORIG_ENCODE_CHAT(teacher_proc, turn["messages"], list(turn["images"]), max_pixels=1048576)
                                kl_term, w_sum = compute_turn_kl_loss_hs(
                                    student_vlm, teacher_vlm, pref_in, t_pref_in, turn["response_ids"], turn["non_coord_mask"], _cw, device, chunk_size=64, use_recompute=True
                                )
                            else:
                                kl_term, w_sum = compute_turn_kl_loss(
                                    student_vlm, teacher_vlm, pref_in, turn["response_ids"], turn["non_coord_mask"], device, chunk_size=64, use_recompute=True
                                )
                        else:
                            kl_term, w_sum = compute_turn_kl_loss(
                                student_vlm, teacher_vlm, pref_in, turn["response_ids"], w_mask, device, chunk_size=64, use_recompute=True
                            )
                        if w_sum > 0.0:
                            traj_kl_sum = kl_term if traj_kl_sum is None else traj_kl_sum + kl_term
                            traj_weight_sum += w_sum
                            batch_tokens += len(turn["response_ids"])

                    if method in ("TK1", "TK2", "TK3", "TKN", "TKV2", "AT", "ATG", "EF", "QK", "QK1", "QKG"):
                        for _a in traj["actions"]:
                            _r = _a.get("tk")
                            if not _r:
                                continue
                            teacher_queries += 1
                            if _r["iou"] is not None:
                                lr_ious.append(_r["iou"])
                            lr_resets.append(1.0 if _r["reset"] else 0.0)
                            if not _r["valid"]:
                                continue
                            teacher_valid_boxes += 1
                            if method in ("ATG", "QK", "QK1", "QKG"):
                                continue
                            if method == "TKV2" and (_r["iou"] is not None and _r["iou"] >= 0.5):
                                continue
                            if method == "AT" and (_r.get("in_at") or (_r.get("cov_t") is not None and _r["cov_t"] >= 0.5)):
                                continue
                            if method == "EF" and _r.get("ef_accept"):
                                continue
                            _kb, _wb = ef_candidate_loss(student_vlm, _r, device) if method == "EF" else tk_branch_kl(student_vlm, teacher_vlm, _r, device)
                            if _wb > 0.0:
                                if method == "EF":
                                    _tk_w = float(_r.get("ef_w", 0.25))
                                elif method in ("TKV2", "AT"):
                                    _tk_w = 0.25
                                else:
                                    _tk_w = 1.0
                                traj_kl_sum = (_tk_w * _kb) if traj_kl_sum is None else traj_kl_sum + (_tk_w * _kb)
                                traj_weight_sum += (_tk_w * _wb)
                                batch_tk_box.append(float(_kb.detach().item()) / _wb)
                    traj_text_loss = (traj_kl_sum / traj_weight_sum) if (traj_kl_sum is not None and traj_weight_sum > 0.0) else None

                    # 3. Coordinate supervision (for B, C, D) on each crop action in trajectory
                    traj_act_loss = None
                    if method in ("B", "C", "D", "M2", "P", "V", "VC") and traj["actions"]:
                        act_terms = []
                        prior_boxes = []
                        for k_act, act in enumerate(traj["actions"]):
                            pre_in = encode_chat(
                                student_proc, act["pre_box_messages"], act["pre_box_images"],
                                suffix=act["pre_box_suffix"], max_pixels=1048576,
                            )
                            t_info = query_teacher_box(teacher_vlm, teacher_proc, pre_in, orig_im, device)
                            teacher_queries += 1
                            if not t_info["valid"]:
                                continue
                            teacher_valid_boxes += 1
                            lr_ious.append(box_iou_float(act["executed_norm_box"], t_info["executed_norm_box"]))
                            if method in ("V", "VC"):
                                _tq = [float(x) for x in t_info["quantized_box"]]
                                if method == "VC" and k_act >= 1 and prior_boxes:
                                    _bt = box_terms(student_vlm, student_proc.tokenizer, pre_in, t_info["quantized_box"],
                                                    coverage_cost_fn(t_info["quantized_box"], prior_boxes), device, "VC", ce_on_digits_only=False)
                                    act_terms.append(_bt["ce"] + GEO.GEO_MU * _bt["cost"])
                                else:
                                    act_terms.append(compute_box_ce_branch(student_vlm, pre_in, t_info["target_suffix_ids"], device))
                                prior_boxes.append([float(x) for x in act["quantized_box"]])
                            elif method == "B":
                                ce_term = compute_box_ce_branch(student_vlm, pre_in, t_info["target_suffix_ids"], device)
                                act_terms.append(ce_term)
                            elif method == "M2":
                                term = m2_term(student_vlm, ref_vlm, student_proc.tokenizer, pre_in, t_info["quantized_box"], act["quantized_box"], device)
                                if term is not None:
                                    act_terms.append(term)
                            elif method == "P":
                                term = p_term(student_vlm, attn_cap, student_proc.tokenizer, pre_in, t_info["quantized_box"], device)
                                if term is not None:
                                    act_terms.append(term)
                            else:
                                # Method C or D: continuous head L_geom against teacher executed normalized box
                                h_last = forward_last_hidden(student_vlm, pre_in, device)[0, -1]
                                b_pred = student_head(h_last)
                                b_targ = torch.tensor(t_info["executed_norm_box"], dtype=torch.float32, device=device)
                                geo_term = compute_geom_loss(b_pred, b_targ)["loss"]
                                act_terms.append(geo_term)
                        if act_terms:
                            traj_act_loss = sum(act_terms) / len(act_terms)

                    if method in ("A", "WH", "HS", "HS5") and traj["actions"]:
                        with torch.no_grad():
                            for act in traj["actions"]:
                                if method in ("WH", "HS", "HS5"):
                                    pre_in = lowres_patch._ORIG_ENCODE_CHAT(teacher_proc, act["pre_box_messages"], list(act["pre_box_images"]),
                                                                            suffix=act["pre_box_suffix"], max_pixels=1048576)
                                else:
                                    pre_in = encode_chat(student_proc, act["pre_box_messages"], act["pre_box_images"],
                                                         suffix=act["pre_box_suffix"], max_pixels=1048576)
                                t_info = query_teacher_box(teacher_vlm, teacher_proc, pre_in, orig_im, device)
                                teacher_queries += 1
                                if t_info["valid"]:
                                    teacher_valid_boxes += 1
                                    lr_ious.append(box_iou_float(act["executed_norm_box"], t_info["executed_norm_box"]))

                    if method in ("G1", "G2"):
                        _gt = row["quantized_box_1000"]
                        _msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                        _pre = encode_chat(student_proc, _msgs, [orig_im], suffix=CANONICAL_BOX_PREFIX, max_pixels=1048576)
                        _bt = box_terms(student_vlm, student_proc.tokenizer, _pre, _gt, gt_cost_fn(_gt), device, method, ce_on_digits_only=True)
                        _g = _bt["ce"] + (GEO.GEO_MU * _bt["cost"] if method == "G2" else 0.0)
                        traj_act_loss = _g if traj_act_loss is None else traj_act_loss + _g

                    total_traj_loss = None
                    if traj_text_loss is not None:
                        total_traj_loss = traj_text_loss
                        batch_kl_losses.append(float(traj_text_loss.detach().item()))
                    if traj_act_loss is not None:
                        weighted_act = lambda_weight * traj_act_loss
                        total_traj_loss = weighted_act if total_traj_loss is None else total_traj_loss + weighted_act
                        batch_act_losses.append(float(traj_act_loss.detach().item()))

                    if total_traj_loss is not None and torch.isfinite(total_traj_loss):
                        (total_traj_loss / eff_batch).backward()

                lora_gn = float(sum(p.grad.float().pow(2).sum().item() for p in lora_params if p.grad is not None) ** 0.5)
                head_gn = float(sum(p.grad.float().pow(2).sum().item() for p in head_params if p.grad is not None) ** 0.5) if head_params else 0.0
                total_gn = float(torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0).item())
                optimizer.step()
                dt = time.monotonic() - t_step

                step_rec = dict(
                    run_id=run_id,
                    method=method,
                    seed=seed,
                    update=upd,
                    loss_text_kl=round(sum(batch_kl_losses) / max(1, len(batch_kl_losses)), 4),
                    loss_action=round(sum(batch_act_losses) / max(1, len(batch_act_losses)), 4) if batch_act_losses else 0.0,
                    lora_grad_norm=round(lora_gn, 4),
                    head_grad_norm=round(head_gn, 4),
                    total_grad_norm=round(total_gn, 4),
                    mean_crops=round(batch_crops / eff_batch, 2),
                    protocol_success_rate=round(batch_proto_ok / eff_batch, 3),
                    scoring_tokens=batch_tokens,
                    teacher_box_queries=teacher_queries,
                    teacher_valid_boxes=teacher_valid_boxes,
                    lr_n_iou=len(lr_ious),
                    lr_mean_iou=round(sum(lr_ious) / max(1, len(lr_ious)), 4),
                    lr_disjoint=round(sum(1 for x in lr_ious if x < 0.2) / max(1, len(lr_ious)), 4),
                    tk_reset_rate=round(sum(lr_resets) / max(1, len(lr_resets)), 4),
                    tk_box_kl=round(sum(batch_tk_box) / max(1, len(batch_tk_box)), 4),
                    qkl_empty_rate=round(QKL_STATS['n_empty'] / max(1, QKL_STATS['n_pos']), 4),
                    qkl_mean_fsize=round(QKL_STATS['sum_fsize'] / max(1, QKL_STATS['n_pos']), 3),
                    qkl_mean_candidates=round(QKL_STATS['n_candidates'] / max(1, QKL_STATS['n_turns']), 2),
                    geo_empty_rate=round(GEO_STATS['n_empty'] / max(1, GEO_STATS['n_pos']), 4),
                    geo_mean_fsize=round(GEO_STATS['sum_fsize'] / max(1, GEO_STATS['n_pos']), 3),
                    geo_infeasible_rate=round(GEO_STATS['n_infeasible'] / max(1, GEO_STATS['n_pos']), 4),
                    geo_format_rate=round(GEO_STATS['n_format'] / max(1, GEO_STATS['n_pos']), 4),
                    geo_unknown_turns=GEO_STATS['n_unknown'],
                    opd_lr=_OPD_LR,
                    peak_mem_gib=round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2),
                    step_sec=round(dt, 2),
                    elapsed_sec=round(time.monotonic() - t_run_start, 1),
                )
                f_log.write(json.dumps(step_rec) + "\n")
                flush_diag(diag_path, upd)
                GEO.flush_diag(RES_DIR / f"train/{run_id}.geo_diag.jsonl", upd)
                f_log.flush()

                if upd % 4 == 0 or upd == 1 or upd in ckpt_steps:
                    log_msg(
                        role,
                        f"[{run_id}] upd {upd}/{total_updates}: kl={step_rec['loss_text_kl']:.4f} "
                        f"act={step_rec['loss_action']:.4f} proto={step_rec['protocol_success_rate']:.2f} "
                        f"crops={step_rec['mean_crops']:.2f} ({dt:.1f}s)",
                    )

                if upd in ckpt_steps:
                    c_dir = run_ckpt_dir / f"step_{upd}"
                    save_student_checkpoint(student_vlm, student_head, c_dir, metadata=step_rec)
                    (c_dir / "READY").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")

        if s0_vlm is not None:
            del s0_vlm, s0_head
        if ref_vlm is not None:
            del ref_vlm
        if attn_cap is not None:
            attn_cap['restore']()
        del optimizer, teacher_vlm, student_vlm, student_head
        torch.cuda.empty_cache()

    if not evaluate:
        return []
    # Evaluate final checkpoint on benchmarks
    log_msg(role, f"=== Evaluating {run_id} (step {total_updates}) on Benchmarks ===")
    eval_vlm, eval_head, eval_proc = load_student_model(
        P_2B, device=device, checkpoint_dir=str(final_ckpt_dir), train_lora=False, train_head=False, seed=seed
    )
    summaries = []
    for b_name in os.environ.get("GEO_BENCHES", "dev_fast_200,vstar,treebench").split(","):
        s = evaluate_model_on_benchmark(
            eval_vlm, eval_head, eval_proc, b_name,
            execution_mode=exec_mode, run_id=run_id, step=total_updates, device=device,
        )
        summaries.append(s)
        log_msg(role, f"[{run_id}][{b_name}] acc={s['answer_accuracy']} cyclic={s['cyclic_accuracy']} mIoU={s['target_miou_all_crops']} joint05={s['joint_iou05']}")
        append_summary_row(s, method=method, seed=seed)

    del eval_vlm, eval_head
    torch.cuda.empty_cache()
    return summaries


SUMMARY_FIELDS = [
    "run_id", "method", "seed", "step", "execution_mode", "benchmark", "visual_budget",
    "n_rows", "n_base_questions", "n_image_groups", "n_correct", "answer_accuracy", "cyclic_accuracy",
    "target_miou_first_crop", "target_miou_all_crops", "target_recall_iou03", "target_recall_iou05",
    "all_targets_hit_iou03", "all_targets_hit_iou05", "joint_iou03", "joint_iou05",
    "crop_precision_iou", "n_with_valid_gt", "n_no_crop", "mean_crops", "protocol_failure_rate", "eval_seconds",
]


def append_summary_row(s: Dict[str, Any], method: str, seed: int):
    row = dict(s, method=method, seed=seed)
    jsonl_p = RES_DIR / "eval/summary_rows.jsonl"
    with jsonl_p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Rebuild summary.csv atomically from summary_rows.jsonl
    all_rows = [json.loads(l) for l in jsonl_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    dedup = {}
    for r in all_rows:
        key = (r["run_id"], r["step"], r["benchmark"], r["visual_budget"])
        dedup[key] = r
    csv_p = RES_DIR / "eval/summary.csv"
    with csv_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in dedup.values():
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("a", "b"), required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--skip-s0-baselines", action="store_true")
    args = parser.parse_args()
    role = args.role
    device = "cuda:0"

    # 1. Wait for Step 2B splits and Step 3-5 checks to finish
    wait_for_file(RES_DIR / "manifests/dataset_manifest.json", role, "Step 2B dataset_manifest.json")
    wait_for_file(RES_DIR / "checks/memory_throughput.json", role, "Step 3-5 memory_throughput.json")

    if role == "a":
        # Role A trains S0, runs Step 7 & 8 diagnostics, then runs B_s42 -> A_s42 -> W_s42 -> B_s17 -> B_s2026
        s0_dir = train_cold_start_s0(role, device=device)
        run_step7_8_diagnostics(role, s0_dir, device=device)
        for run_id, method, seed in [
            ("B_s42", "B", 42),
            ("A_s42", "A", 42),
            ("W_s42", "W", 42),
            ("B_s17", "B", 17),
            ("B_s2026", "B", 2026),
        ]:
            run_opd_branch(role, run_id, method, seed, s0_dir, total_updates=args.updates, device=device)

        # Follow up with HR-Bench 4K / 8K on Role A checkpoints
        for run_id, method, seed in [("B_s42", "B", 42), ("A_s42", "A", 42), ("W_s42", "W", 42)]:
            ckpt_dir = CKPT_ROOT / run_id / f"step_{args.updates}"
            eval_vlm, eval_head, eval_proc = load_student_model(
                P_2B, device=device, checkpoint_dir=str(ckpt_dir), train_lora=False, train_head=False, seed=seed
            )
            for b_name in ["hrbench_4k", "hrbench_8k"]:
                s = evaluate_model_on_benchmark(
                    eval_vlm, eval_head, eval_proc, b_name,
                    execution_mode="text", run_id=run_id, step=args.updates, device=device,
                )
                append_summary_row(s, method=method, seed=seed)
            del eval_vlm, eval_head
            torch.cuda.empty_cache()
    else:
        # Role B: While Role A trains S0, Role B evaluates frozen Teacher T on the benchmarks first!
        log_msg(role, "=== Evaluating Frozen Teacher T (Qwen3-VL-8B-Instruct) on dev_fast_200, V*, TreeBench ===")
        t_vlm, t_proc = load_teacher_model(P_8B, device=device)
        dummy_head = OrderedBoxHead(4096).to(device).eval()
        for b_name in ["dev_fast_200", "vstar", "treebench"]:
            s = evaluate_model_on_benchmark(
                t_vlm, dummy_head, t_proc, b_name,
                execution_mode="text", run_id="T_teacher_8b", step=0, device=device,
            )
            log_msg(role, f"[T_teacher_8b][{b_name}] acc={s['answer_accuracy']} mIoU={s['target_miou_all_crops']} joint05={s['joint_iou05']}")
            append_summary_row(s, method="T", seed=0)
        del t_vlm, dummy_head
        torch.cuda.empty_cache()

        # Wait for S0/READY from Role A
        s0_dir = CKPT_ROOT / "S0"
        wait_for_file(s0_dir / "READY", role, "Cold-start S0 checkpoint")

        # Evaluate S0_T and S0_H baselines on dev_fast_200, vstar, treebench
        s0_vlm, s0_head, s0_proc = load_student_model(
            P_2B, device=device, checkpoint_dir=str(s0_dir), train_lora=False, train_head=False, seed=42
        )
        if args.skip_s0_baselines:
            # S0_T is already on disk; sanity-check the re-warmed head on the dev split before Method C
            s0_plan = [("S0_H", "continuous_head", ["dev_fast_200"])]
        else:
            s0_plan = [
                ("S0_T", "text", ["dev_fast_200", "vstar", "treebench"]),
                ("S0_H", "continuous_head", ["dev_fast_200", "vstar", "treebench"]),
            ]
        for s0_run_id, s0_mode, s0_benches in s0_plan:
            for b_name in s0_benches:
                s = evaluate_model_on_benchmark(
                    s0_vlm, s0_head, s0_proc, b_name,
                    execution_mode=s0_mode, run_id=s0_run_id, step=0, device=device,
                )
                log_msg(role, f"[{s0_run_id}][{b_name}] acc={s['answer_accuracy']} mIoU={s['target_miou_all_crops']} joint05={s['joint_iou05']} proto_fail={s['protocol_failure_rate']}")
                append_summary_row(s, method=s0_run_id, seed=42)
        del s0_vlm, s0_head
        torch.cuda.empty_cache()

        run_opd_branch(role, "C_s42", "C", 42, s0_dir, total_updates=args.updates, device=device)

        if args.skip_s0_baselines:
            # Complete the retrained-head S0_H baseline right after the core Method C run
            s0_vlm, s0_head, s0_proc = load_student_model(
                P_2B, device=device, checkpoint_dir=str(s0_dir), train_lora=False, train_head=False, seed=42
            )
            for b_name in ["vstar", "treebench"]:
                s = evaluate_model_on_benchmark(
                    s0_vlm, s0_head, s0_proc, b_name,
                    execution_mode="continuous_head", run_id="S0_H", step=0, device=device,
                )
                log_msg(role, f"[S0_H][{b_name}] acc={s['answer_accuracy']} mIoU={s['target_miou_all_crops']} joint05={s['joint_iou05']}")
                append_summary_row(s, method="S0_H", seed=42)
            del s0_vlm, s0_head
            torch.cuda.empty_cache()

        # Run D_s42 -> H0_s42 -> C_s17 -> C_s2026
        for run_id, method, seed in [
            ("D_s42", "D", 42),
            ("H0_s42", "H0", 42),
            ("C_s17", "C", 17),
            ("C_s2026", "C", 2026),
        ]:
            run_opd_branch(role, run_id, method, seed, s0_dir, total_updates=args.updates, device=device)

        # Follow up with HR-Bench 4K / 8K on Role B checkpoints
        for run_id, method, seed in [("C_s42", "C", 42), ("D_s42", "D", 42), ("H0_s42", "H0", 42)]:
            ckpt_dir = CKPT_ROOT / run_id / f"step_{args.updates}"
            eval_vlm, eval_head, eval_proc = load_student_model(
                P_2B, device=device, checkpoint_dir=str(ckpt_dir), train_lora=False, train_head=False, seed=seed
            )
            for b_name in ["hrbench_4k", "hrbench_8k"]:
                s = evaluate_model_on_benchmark(
                    eval_vlm, eval_head, eval_proc, b_name,
                    execution_mode="continuous_head", run_id=run_id, step=args.updates, device=device,
                )
                append_summary_row(s, method=method, seed=seed)
            del eval_vlm, eval_head
            torch.cuda.empty_cache()

    log_msg(role, f"=== ALL OVERNIGHT RUNS FOR ROLE {role.upper()} COMPLETED! ===")


if __name__ == "__main__":
    main()
