#!/usr/bin/env python3
"""
Diffractive plate vs DMD on the thesis two-layer scalar operators, reconstructed with the current solver.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import scico.numpy as snp

from pipeline.reconstruct import run_pnp, PRIOR_PRESETS as PRESETS
from pipeline.metrics import axial_profile, depth_metrics
from pipeline import viz


def add_noise_with_snr(clean, snr_db, d_size, n_pattern):
    """Verbatim FYP libs/obj_utils/noise_sim.add_noise_with_snr."""
    clean_power = torch.mean(clean.view(n_pattern, d_size * d_size))
    noise_power = clean_power / (10 ** (snr_db / 10))
    noise = (torch.randn_like(clean.view(n_pattern, d_size * d_size)).T * noise_power).T
    return clean + noise.flatten()


def load_op(d, strategy, data_type, dsize, npattern, point, snr):
    A = torch.load(d / strategy / f"A_{dsize}_{npattern}.pt", map_location="cpu")
    A = A["matrix"] if isinstance(A, dict) else A
    Xr = torch.load(d / strategy / f"Xr_{data_type}.pt", map_location="cpu")
    Y = torch.load(d / strategy / f"Y_{data_type}_{dsize}_{npattern}.pt", map_location="cpu")
    gt = Xr[point].reshape(8, 32, 32).float().numpy()
    y = add_noise_with_snr(Y[point].float(), snr, dsize, npattern)
    return A.float(), gt, y


def main() -> int:
    ap = argparse.ArgumentParser(description="D2NN vs DMD on the thesis operators.")
    ap.add_argument("--data", default="backup_fyp")
    ap.add_argument("--strategies", default="D2NN,DMD")
    ap.add_argument("--data_type", default="neural_cell")
    ap.add_argument("--dsize", type=int, default=8)
    ap.add_argument("--npattern", type=int, default=128)
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--compressions", default="1,8")
    ap.add_argument("--preset", default="dncnn")
    ap.add_argument("--out", default="results/recon/fyp_compare")
    args = ap.parse_args()

    d = Path(args.data)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    comps = [float(c) for c in args.compressions.split(",")]
    prior = PRESETS.get(args.preset, {})
    stamp = time.strftime("%Y%m%d_%H%M%S"); out = Path(args.out) / stamp; out.mkdir(parents=True, exist_ok=True)

    ops = {s: load_op(d, s, args.data_type, args.dsize, args.npattern, args.point, args.snr) for s in strategies}
    gt = ops[strategies[0]][1]; gt_j = snp.array(gt); gt_std = float(axial_profile(gt).std())
    obj_shape = (8, 32, 32); n_vox = 8192
    print(f"data={args.data_type} point={args.point} d_size={args.dsize} n_pattern={args.npattern} "
          f"preset={args.preset} strategies={strategies} GT axial-std={gt_std:.4f}")

    for c in comps:
        vols = {"GT": gt}; profs = {}
        for s in strategies:
            A, _, y = ops[s]
            nm = min(A.shape[0], int(round(n_vox / c)))
            est, _ = run_pnp(snp.array(A.numpy()), snp.array(y.numpy()),
                             obj_shape=obj_shape, gt=gt_j, n_measure=nm, **prior)
            est = np.asarray(est); m = depth_metrics(gt, est)
            sh = float(axial_profile(est).std() / (gt_std + 1e-12))
            print(f"  c={c:g} nm={nm:5d}  {s:6s}: psnr={m['psnr']:.2f}dB  "
                  f"centroid_err={m['centroid_err']:.3f}  profile_l1={m['profile_l1']:.4f}  shape={sh:.2f}")
            vols[s] = est; profs[s] = axial_profile(est)
        viz.show_mips(vols, title=f"FYP {args.data_type} pt{args.point} c={c:g} ({args.preset})").savefig(
            out / f"mips_c{c:g}.png", dpi=140, bbox_inches="tight")
        viz.overlay_axial_profile(axial_profile(gt), profs,
                                  title=f"D2NN vs DMD — {args.data_type} pt{args.point} c={c:g} ({args.preset})").savefig(
            out / f"profile_c{c:g}.png", dpi=140, bbox_inches="tight")
    print(f"artifacts in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
