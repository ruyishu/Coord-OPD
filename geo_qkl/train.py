"""Geo-QKL training and evaluation entry point.
Usage:
    python -m geo_qkl.train <run_id> <updates> <seed> <eval|noeval>
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import run_opd_branch


def main():
    if len(sys.argv) < 5:
        print("Usage: python -m geo_qkl.train <run_id> <updates> <seed> <eval|noeval>")
        sys.exit(1)
    run_id, updates, seed, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    assert mode in ("eval", "noeval")
    ws = Path(os.environ.get("COORD_OPD_WORKSPACE", str(HERE.parent)))
    run_opd_branch(
        "b",
        run_id,
        "QKG",
        seed,
        ws / "checkpoints" / "S0",
        total_updates=updates,
        device=os.environ.get("CUDA_DEVICE", "cuda:0"),
        evaluate=(mode == "eval"),
    )
    print("RUN_DONE", run_id, flush=True)


if __name__ == "__main__":
    main()
