#!/usr/bin/env python3
"""
Check the vectorial excitation field against its scalar limit at low NA.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coder.psf import psf_model


def fields(m, device="cpu"):
    """Return (vector_intensity, scalar_intensity) each (Nx, Ny, Nz) from one psf_model."""
    F0 = m.Func0.to(device); F1 = m.Func1.to(device); F2 = m.Func2.to(device)
    THETA = m.THETA.to(device); theta = m.theta.to(device); Phi = m.Phi.to(device)
    cos2phi = torch.cos(2 * Phi); sin2phi = torch.sin(2 * Phi); cosphi = torch.cos(Phi)
    Iv = torch.zeros(m.Nx, m.Ny, m.Nz); Is = torch.zeros(m.Nx, m.Ny, m.Nz)
    for k in range(len(m.U)):
        F3 = torch.exp(-1j * m.U[k] * torch.cos(THETA)).to(device)
        I0 = torch.trapz(F0 * F3, theta, dim=2)
        I1 = torch.trapz(F1 * F3, theta, dim=2)
        I2 = torch.trapz(F2 * F3, theta, dim=2)
        Ex = 1j * m.A * (I0 + I2 * cos2phi)
        Ey = 1j * m.A * I2 * sin2phi
        Ez = -2 * m.A * I1 * cosphi
        Iv[:, :, k] = (Ex.abs() ** 2 + Ey.abs() ** 2 + Ez.abs() ** 2).cpu()
        Is[:, :, k] = ((1j * m.A * I0).abs() ** 2).cpu()          # scalar = I0 term only
    return Iv, Is


def compare(Iv, Is):
    a = (Iv / Iv.max()).flatten().numpy(); b = (Is / Is.max()).flatten().numpy()
    corr = float(np.corrcoef(a, b)[0, 1])
    relerr = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    return corr, relerr


def main() -> int:
    ap = argparse.ArgumentParser(description="Vector-vs-scalar excitation field validation.")
    ap.add_argument("--nas", default="0.1,0.2,0.4,0.8")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--pitch", type=float, default=0.125)
    ap.add_argument("--dz", type=float, default=0.25)
    ap.add_argument("--lambda_um", type=float, default=0.532)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/coder/verify_field")
    args = ap.parse_args()

    nas = [float(x) for x in args.nas.split(",")]
    dev = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cy = args.N // 2

    fig, ax = plt.subplots(len(nas), 3, figsize=(9, 2.6 * len(nas)), squeeze=False)
    print(f"grid N={args.N} pitch={args.pitch}um nz={args.nz} dz={args.dz}um  device={dev}")
    print(f"{'NA':>6} {'corr':>8} {'rel_err':>9}   (corr~1 & rel_err~0 => scalar==vector)")
    for r, na in enumerate(nas):
        m = psf_model(na, 1.0, args.lambda_um, args.pitch, args.pitch, args.dz, args.N, args.N, args.nz)
        Iv, Is = fields(m, dev)
        corr, relerr = compare(Iv, Is)
        print(f"{na:6.2f} {corr:8.5f} {relerr:9.4f}")
        xz_v = Iv[:, cy, :].t().numpy(); xz_s = Is[:, cy, :].t().numpy()  # (nz, N)
        vmax = max(xz_v.max(), xz_s.max())
        for c, (img, ttl) in enumerate(((xz_s, "scalar |I0|^2"), (xz_v, "vector |E|^2"),
                                        (np.abs(xz_v / vmax - xz_s / vmax), "|diff| (norm)"))):
            im = ax[r, c].imshow(img, cmap="inferno", aspect="auto")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(ttl, fontsize=9)
        ax[r, 0].set_ylabel(f"NA {na:g}\ncorr {corr:.4f}", fontsize=9)
    fig.suptitle("Excitation field: scalar (I0) vs vector (Richards-Wolf), XZ cross-section")
    fig.tight_layout(); fig.savefig(out / "field_scalar_vs_vector.png", dpi=140, bbox_inches="tight")
    print(f"figure -> {out/'field_scalar_vs_vector.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
