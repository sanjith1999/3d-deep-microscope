#!/usr/bin/env python3
"""
Designed scalar coder vs random plate vs DMD on prebuilt operators at a common column energy.
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

from pipeline.forward import add_noise_snr
from pipeline.reconstruct import run_pnp, PRIOR_PRESETS as PRESETS
from pipeline.metrics import axial_profile, depth_metrics
from pipeline import viz

# label -> (path, is_dict)
DEFAULT_OPS = {
    "designed":    ("results/recon/scalar_A_designed.pt", False),
    "random-D2NN": ("results/recon/scalar_A_p128_seed5.pt", False),
    "DMD":         ("backup_fyp/DMD/A_8_128.pt", True),
}


def load_A(path, is_dict, target_col=27.0):
    A = torch.load(path, map_location="cpu")
    A = (A["matrix"] if is_dict else A).float()
    ce = float(A.norm(dim=0).mean())
    return A / (ce + 1e-12) * target_col, ce, A.shape           # normalise to common col-energy


def main() -> int:
    ap = argparse.ArgumentParser(description="designed vs random vs DMD via DnCNN.")
    ap.add_argument("--xr", default="backup_fyp/D2NN/Xr_neural_cell.pt")
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--compressions", default="1,8")
    ap.add_argument("--target_col", type=float, default=27.0)
    ap.add_argument("--out", default="results/recon/designed_compare")
    args = ap.parse_args()

    Xr = torch.load(args.xr, map_location="cpu")
    gt = Xr[args.point].reshape(8, 32, 32).float(); gt_np = gt.numpy()
    gt_j = snp.array(gt_np); gt_std = float(axial_profile(gt_np).std())
    stamp = time.strftime("%Y%m%d_%H%M%S"); out = Path(args.out) / stamp; out.mkdir(parents=True, exist_ok=True)

    ops = {}
    for name, (path, is_dict) in DEFAULT_OPS.items():
        A, ce, shp = load_A(path, is_dict, args.target_col)
        ops[name] = A
        print(f"[{name:12s}] {path}  shape={tuple(shp)}  raw col-energy={ce:.3e} -> normalised {args.target_col}")

    n_vox = 8192
    for c in [float(x) for x in args.compressions.split(",")]:
        print(f"\n--- c={c:g} ---")
        for pname in ("dncnn", "l1"):
            prior = PRESETS[pname]; vols = {"GT": gt_np}; profs = {}
            for name, A in ops.items():
                nm = min(A.shape[0], int(round(n_vox / c)))
                Y = add_noise_snr(A @ gt.reshape(-1), args.snr, seed=args.point)
                est, _ = run_pnp(snp.array(A.numpy()), snp.array(Y.numpy()),
                                 obj_shape=(8, 32, 32), gt=gt_j, n_measure=nm, **prior)
                est = np.asarray(est); m = depth_metrics(gt_np, est)
                sh = float(axial_profile(est).std() / (gt_std + 1e-12))
                print(f"  {pname:5s} {name:12s}: psnr={m['psnr']:.2f}dB  centroid_err={m['centroid_err']:.3f}  "
                      f"profile_l1={m['profile_l1']:.4f}  shape={sh:.2f}")
                vols[name] = est; profs[name] = axial_profile(est)
            viz.show_mips(vols, title=f"{pname} c={c:g}: designed vs random vs DMD").savefig(
                out / f"mips_{pname}_c{c:g}.png", dpi=140, bbox_inches="tight")
            viz.overlay_axial_profile(axial_profile(gt_np), profs,
                                      title=f"{pname} c={c:g}: designed vs random vs DMD").savefig(
                out / f"profile_{pname}_c{c:g}.png", dpi=140, bbox_inches="tight")
    print(f"\nartifacts in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
