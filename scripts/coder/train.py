#!/usr/bin/env python3
"""
Train one surrogate-objective coder and write its figures.

    python scripts/coder/train.py
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_config(path: str | None):
    if path is None:
        from configs.default import get_config
        return get_config()
    spec = importlib.util.spec_from_file_location("_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a surrogate-objective diffractive coder.")
    ap.add_argument("--config", default=None, help="path to a .py exposing get_config()")
    ap.add_argument("--steps", type=int, default=None, help="override training num_steps")
    ap.add_argument("--device", default=None, help="override device (cuda/cpu)")
    ap.add_argument("--target", default=None, help="bimodal | exponential | soft_binary")
    ap.add_argument("--out", default="results/coder", help="override output root dir")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.training["num_steps"] = args.steps
    if args.device is not None:
        cfg.device = args.device
    if args.target is not None:
        cfg.target["name"] = args.target
    if args.out is not None:
        cfg.out_dir = args.out
    if args.seed is not None:
        cfg.seed = args.seed

    from coder.train import train
    from coder.figures import all_figures

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.out_dir) / f"{stamp}_{cfg.target['name']}"
    print("=" * 70)
    print(f"FRIS coder design | NA_ex={cfg.na_ex} target={cfg.target['name']} "
          f"device={cfg.device} fine={cfg.n_fine}@{cfg.pitch_fine}um->coarse={cfg.n_fine // cfg.bin_factor}@{cfg.pitch_fine * cfg.bin_factor}um "
          f"layers={cfg.n_layers} planes={cfg.nz} patterns={cfg.n_alphas * cfg.n_thetas} "
          f"steps={cfg.training['num_steps']}")
    print(f"output -> {run_dir}")
    print("=" * 70)

    result = train(cfg, run_dir)
    all_figures(result)

    print("-" * 70)
    print(f"DONE in {result['manifest']['wall_seconds']:.1f}s | "
          f"final axial coherence = {result['manifest']['final_coherence']:.4f} (lower=better)")
    print(f"artifacts in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
