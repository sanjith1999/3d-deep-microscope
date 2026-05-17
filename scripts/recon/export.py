#!/usr/bin/env python3
"""
End-to-end sanity export: one synthetic object through the forward operator and the PnP solver.
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

from pipeline.forward import ExportForward, emission_psf, spherical_object, reduce_object, add_noise_snr
from pipeline.reconstruct import run_pnp


def main() -> int:
    ap = argparse.ArgumentParser(description="PnP export sanity check.")
    ap.add_argument("--h2", required=True, help="designed illumination volume (speckle_volume.pt)")
    ap.add_argument("--patterns", type=int, default=8, help="# patterns (8 => n_measure=8192, c=1)")
    ap.add_argument("--d_size", type=int, default=4)
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--radius", type=float, default=3.0, help="sphere radius (um)")
    ap.add_argument("--out", default="results/recon/export_sanity")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    H2_full = torch.load(args.h2, map_location=dev)              # (T, nz, N, N)
    T, nz, N, _ = H2_full.shape
    H2 = H2_full[:args.patterns]
    print(f"H2 {tuple(H2_full.shape)} -> using {args.patterns} patterns | grid nz={nz} N={N} d_size={args.d_size}")

    emPSF = emission_psf(nz, N, dz_um=0.25, pitch_um=0.25, na_em=0.1, device=dev)
    fwd = ExportForward(H2, emPSF, patch=(4, 4, 4), d_size=args.d_size, device=dev)
    print(f"recon grid {fwd.recon_shape}  measurements {fwd.meas_shape} = {T * fwd.Nd * fwd.Nd}")

    t = time.time(); A = fwd.build_A(); print(f"A {tuple(A.shape)} built in {time.time()-t:.0f}s")

    Xa = spherical_object(nz, N, 0.25, 0.25, radius_um=args.radius, device=dev)
    X_r = reduce_object(Xa, (4, 4, 4))                           # (rz, rx, ry)
    Y = add_noise_snr(A @ X_r.reshape(-1), args.snr)
    print(f"object occupancy(recon) = {(X_r > 0.05).float().mean().item():.3f}")

    A_j, Y_j, gt_j = (snp.array(t_.detach().cpu().numpy()) for t_ in (A, Y, X_r))
    t = time.time()
    est, psnr = run_pnp(A_j, Y_j, obj_shape=tuple(X_r.shape), gt=gt_j, verbose=True)
    dt = time.time() - t
    print("-" * 60)
    print(f"recon in {dt:.0f}s | PSNR_3D = {psnr:.2f} dB")

    # Figure: GT vs reconstruction across depth.
    gt = np.asarray(gt_j); rec = np.asarray(est); rz = gt.shape[0]
    fig, ax = plt.subplots(2, rz, figsize=(2.0 * rz, 4.2))
    vmax = float(gt.max())
    for k in range(rz):
        ax[0, k].imshow(gt[k], cmap="inferno", vmin=0, vmax=vmax); ax[0, k].axis("off")
        ax[0, k].set_title(f"z{k}", fontsize=8)
        ax[1, k].imshow(rec[k], cmap="inferno", vmin=0, vmax=vmax); ax[1, k].axis("off")
    ax[0, 0].set_ylabel("GT", fontsize=10); ax[1, 0].set_ylabel("recon", fontsize=10)
    fig.suptitle(f"Export sanity — designed coder, {args.patterns} patterns, SNR {args.snr:.0f} dB — PSNR3D {psnr:.2f} dB")
    fig.tight_layout(); fig.savefig(run_dir / "sanity.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    manifest = {"h2": args.h2, "patterns": args.patterns, "d_size": args.d_size, "snr_db": args.snr,
                "radius_um": args.radius, "recon_shape": list(fwd.recon_shape),
                "n_measure": A.shape[0], "psnr3d": psnr, "recon_seconds": dt,
                "A_build_seconds": None}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"artifacts in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
