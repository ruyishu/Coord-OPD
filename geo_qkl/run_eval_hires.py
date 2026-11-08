"""High-resolution single-view reference: evaluate a checkpoint at 1MP with cropping disabled.

  python run_eval_hires.py <tag> <S0|TEACHER|run/step_N> <bench> [limit]
Uses the stock evaluator and scorer; only the rollout is forced to max_crops=0.
"""
import json
import os
import sys
from pathlib import Path

WS = Path("/data/ibu-bi-ai-share-disk/zk/project/coord-opd/v1_workspace")
BASE = WS / "probe10_20261028"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(WS))

tag, ckpt, bench = sys.argv[1:4]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.environ.pop("LOWRES_OV_PIXELS", None)

import v1_lowres_pipeline as P  # noqa: E402
from v1_core import OrderedBoxHead, load_student_model, load_teacher_model  # noqa: E402

_orig_rollout = P.rollout_trajectory


def _single_view_rollout(*args, **kwargs):
    kwargs["max_crops"] = 0
    return _orig_rollout(*args, **kwargs)


P.rollout_trajectory = _single_view_rollout

dev = "cuda:0"
step = int(ckpt.rsplit("step_", 1)[1]) if "step_" in ckpt else 0
if ckpt == "TEACHER":
    vlm, proc = load_teacher_model(P.P_8B, device=dev)
    head = OrderedBoxHead(4096).to(dev).eval()
else:
    vlm, head, proc = load_student_model(P.P_2B, device=dev, checkpoint_dir=str(WS / "checkpoints" / ckpt),
                                         train_lora=False, train_head=False, seed=42)

s = P.evaluate_model_on_benchmark(vlm, head, proc, bench, execution_mode="text", run_id=tag, step=step,
                                  device=dev, limit=limit)
row = dict(s, tag=tag, ckpt=ckpt, ov=1048576, single_view=True)
with (BASE / "lowres/eval_rows.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("EVAL_DONE", tag, ckpt, bench, "acc", s["answer_accuracy"], "n", s["n_rows"], "crops", s["mean_crops"],
      "nocrop", s["n_no_crop"], "protofail", s["protocol_failure_rate"], flush=True)
