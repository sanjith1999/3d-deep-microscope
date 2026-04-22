#!/usr/bin/env python3
"""
Depth-correlation matrices of the DMD, random-plate and designed coders (symmetry breaking).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from configs.default import get_config
from coder import baselines
from coder.figures import symmetry_comparison


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 baseline coder comparison (vectorial model).")
    ap.add_argument("--designed", required=True, help="path to a trained coder_final.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0, help="seed for the DMD / random-D2NN coders")
    ap.add_argument("--out", default="results/coder/e1")
    args = ap.parse_args()

    cfg = get_config()
    if args.device is not None:
        cfg.device = args.device
    focus = cfg.nz // 2 - 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"E1 baselines (vectorial) | device={cfg.device} coarse={cfg.n_fine // cfg.bin_factor}"
          f"@{cfg.pitch_fine * cfg.bin_factor}um planes={cfg.nz} patterns={cfg.n_alphas * cfg.n_thetas} "
          f"NA_ex={cfg.na_ex} focus_idx={focus}")
    print(f"designed coder: {args.designed}")
    print(f"output -> {run_dir}")
    print("=" * 70)

    scores: dict[str, dict] = {}

    def add(name: str, vol: torch.Tensor) -> None:
        scores[name] = baselines.score_volume(vol, focus, cfg.dz_um)
        del vol
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

    t0 = time.time()
    add("DMD", baselines.dmd_volume(cfg, seed=args.seed))
    add("random-D2NN", baselines.random_d2nn_volume(cfg, seed=args.seed))
    add("designed", baselines.designed_volume(cfg, args.designed))
    wall = time.time() - t0

    print("-" * 70)
    print(f"{'coder':<14}{'sym(+d/-d)':>12}{'off-diag':>10}{'coh(raw)':>10}{'coh@res':>9}")
    for name, s in scores.items():
        print(f"{name:<14}{s['sym']:>12.4f}{s['offdiag']:>10.4f}"
              f"{s['coherence']:>10.4f}{s['coherence_res']:>9.4f}")
    print(f"(coh@res = max |corr| beyond {scores['designed']['res_sep']} planes ~= 1 um; lower=better)")

    symmetry_comparison(scores, cfg.dz_um, run_dir / "comparison.png")

    summary = {
        "config": {"n_coarse": cfg.n_fine // cfg.bin_factor, "dz_um": cfg.dz_um, "nz": cfg.nz,
                   "n_patterns": cfg.n_alphas * cfg.n_thetas, "na_ex": cfg.na_ex,
                   "focus_idx": focus, "designed": args.designed, "seed": args.seed},
        "wall_seconds": wall,
        "metrics": {name: {k: s[k] for k in ("sym", "offdiag", "coherence", "coherence_res", "res_sep")}
                    for name, s in scores.items()},
        "profiles": {name: {"seps": s["seps"].tolist(), "max_c": s["max_c"].tolist(),
                            "mean_c": s["mean_c"].tolist()} for name, s in scores.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"figure + summary in: {run_dir}  ({wall:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
