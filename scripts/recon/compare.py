#!/usr/bin/env python3
"""
Bead-phantom reconstruction: designed vs random plate vs DMD under identical dose and solver.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scico.numpy as snp

from configs.default import get_config
from coder import baselines
from pipeline.forward import ExportForward, emission_psf, bead_phantom, reduce_object, add_noise_snr
from pipeline.reconstruct import run_pnp

BEADS = [(1, 10, 10), (2, 22, 10), (4, 16, 16), (5, 10, 22), (6, 22, 22)]   # (rz, rx, ry)


def main() -> int:
    ap = argparse.ArgumentParser(description="Designed vs DMD vs random reconstruction comparison.")
    ap.add_argument("--designed", required=True, help="designed illumination (speckle_volume.pt)")
    ap.add_argument("--patterns", type=int, default=32)
    ap.add_argument("--d_size", type=int, default=4)
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object", default=None, help="path to a reduced object Xr (rz,rx,ry); default = bead phantom")
    ap.add_argument("--out", default="results/recon/compare")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = get_config(); cfg.device = args.device
    dev, T = args.device, args.patterns
    nz, N = cfg.nz, cfg.n_fine // cfg.bin_factor
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / stamp; run_dir.mkdir(parents=True, exist_ok=True)

    print(f"grid nz={nz} N={N} d_size={args.d_size} patterns={T} snr={args.snr} beads={len(BEADS)}")
    H2 = {
        "designed": torch.load(args.designed, map_location=dev)[:T],
        "random-D2NN": baselines.random_d2nn_volume(cfg, seed=args.seed)[:T],
        "DMD": baselines.dmd_volume(cfg, seed=args.seed)[:T],
    }
    emPSF = emission_psf(nz, N, 0.25, 0.25, cfg.na_em, device=dev)
    if args.object:
        Xr_raw = torch.load(args.object, map_location=dev).float()
        if Xr_raw.ndim == 2:            # a batch of flattened objects -> take the first
            Xr_raw = Xr_raw[0]
        X_r = Xr_raw.reshape(nz // 4, N // 4, N // 4)
        X_r = X_r / (X_r.max() + 1e-12)
        print(f"object: {args.object}  shape {tuple(X_r.shape)}")
    else:
        X_r = reduce_object(bead_phantom(nz, N, BEADS, radius_um=1.0, device=dev))
    gt_j = snp.array(X_r.detach().cpu().numpy())

    order = ["designed", "random-D2NN", "DMD"]
    res = {}
    for name in order:
        fwd = ExportForward(H2[name], emPSF, patch=(4, 4, 4), d_size=args.d_size, device=dev)
        t = time.time(); A = fwd.build_A(); bt = time.time() - t
        Y = add_noise_snr(A @ X_r.reshape(-1), args.snr, seed=args.seed)
        est, psnr = run_pnp(snp.array(A.detach().cpu().numpy()), snp.array(Y.detach().cpu().numpy()),
                            obj_shape=tuple(X_r.shape), gt=gt_j)
        res[name] = {"psnr": psnr, "est": np.asarray(est), "A_build_s": bt}
        print(f"{name:14s}: A build {bt:.0f}s | PSNR3D {psnr:.2f} dB")
        del A, fwd; torch.cuda.empty_cache()

    # Figure: rows = GT, designed, random, DMD ; cols = depth planes.
    gt = np.asarray(gt_j); rz = gt.shape[0]; vmax = float(gt.max())
    rows = [("GT", gt)] + [(n, res[n]["est"]) for n in order]
    fig, ax = plt.subplots(len(rows), rz, figsize=(1.9 * rz, 1.9 * len(rows)))
    for r, (label, vol) in enumerate(rows):
        for k in range(rz):
            ax[r, k].imshow(vol[k], cmap="inferno", vmin=0, vmax=vmax); ax[r, k].axis("off")
            if r == 0:
                ax[r, k].set_title(f"z{k}", fontsize=8)
        tag = label if r == 0 else f"{label}  {res[label]['psnr']:.1f} dB"
        ax[r, 0].text(-0.35, 0.5, tag, transform=ax[r, 0].transAxes, rotation=90,
                      va="center", ha="center", fontsize=9)
    fig.suptitle(f"Depth recovery: designed vs random-D2NN vs DMD — {T} patterns, SNR {args.snr:.0f} dB, matched dose")
    fig.tight_layout(); fig.savefig(run_dir / "compare.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    summary = {"patterns": T, "d_size": args.d_size, "snr_db": args.snr, "seed": args.seed,
               "beads_recon_zxy": BEADS, "designed": args.designed,
               "psnr3d": {n: res[n]["psnr"] for n in order},
               "A_build_seconds": {n: res[n]["A_build_s"] for n in order}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("-" * 60)
    print("PSNR3D:", {n: round(res[n]["psnr"], 2) for n in order})
    print(f"artifacts in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
