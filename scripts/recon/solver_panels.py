#!/usr/bin/env python3
"""
Side-by-side panels: baseline PnP vs learned-prior PnP on the same measurements.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import numpy as np
import scico.metric  # noqa: F401  (pipeline.metrics uses scico.metric.psnr)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.metrics import score
from pipeline.metrics_paper import ssim3d_foreground

P = 0.64


def depth_rgb(v, vmax, cm):
    z = v.argmax(0) / (v.shape[0] - 1); a = np.clip(v.max(0) / vmax, 0, 1)
    return cm(z)[..., :3] * a[..., None]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--specs", default="neural_cell_p384,neural_cell,neural_cell_p973")
    a = ap.parse_args()
    d = _ROOT / "results/paper/solver_lab/random1L_gap66/volumes"; out = d.parent / "panels"; out.mkdir(exist_ok=True)
    cm = plt.get_cmap("viridis")
    cols = [("paper solver, 256 frames", "base", 256), ("paper solver, 32 frames (8x)", "base", 32),
            ("learned solver, 32 frames (8x)", "t23n", 32), ("learned solver, 256 frames", "t23n", 256)]
    for spec in a.specs.split(","):
        vols = [("reference", None)]
        gt = None
        for title, cfg, k in cols:
            z = np.load(d / f"{cfg}_{spec}_k{k}_s0.npz"); gt = z["gt"]; e = z["est"]
            m = score(gt, e); fg = ssim3d_foreground(gt, e)
            vols.append((f"{title}\n{m['psnr']:.1f} dB | depth {m['local_depth_err'] * P:.2f} um | fg-SSIM {fg:.2f}", e))
        vols[0] = ("reference", gt)
        vmax = float(max(v.max() for _, v in vols)); n = len(vols)
        fig, ax = plt.subplots(3, n, figsize=(2.4 * n, 5.4), gridspec_kw={"height_ratios": [1, 1, 16 / 48]})
        ext_xy = [0, 48 * P, 48 * P, 0]; ext_xz = [0, 48 * P, 16 * P, 0]
        for j, (t, v) in enumerate(vols):
            ax[0, j].imshow(depth_rgb(v, vmax, cm), extent=ext_xy)
            ax[1, j].imshow(v.max(0), cmap="magma", vmin=0, vmax=vmax, extent=ext_xy)
            ax[2, j].imshow(v.max(2), cmap="magma", vmin=0, vmax=vmax, extent=ext_xz)
            ax[0, j].set_title(t, fontsize=6)
            for r in range(3):
                ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        ax[0, 0].set_ylabel("xy MIP, colour = depth", fontsize=6); ax[1, 0].set_ylabel("xy MIP", fontsize=6)
        ax[2, 0].set_ylabel("xz MIP", fontsize=6)
        for r, (e, yf) in ((1, (ext_xy, 0.93)), (2, (ext_xz, 0.85))):
            ax[r, 0].plot([1, 6], [e[2] * yf] * 2, color="w", lw=1.2)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 16 * P)); cax = fig.add_axes([0.92, 0.66, 0.01, 0.2])
        cb = fig.colorbar(sm, cax=cax); cb.set_label("depth (um)", fontsize=6); cb.ax.tick_params(labelsize=5)
        fig.subplots_adjust(right=0.9, wspace=0.08, hspace=0.12)
        fig.suptitle(f"{spec}: same one-plate measurements (noise seed 0, draw 0), paper PnP (DnCNN-17M) vs learned 2D+3D "
                     f"denoisers; shared limits [0, {vmax:.2f}]; 5 um bar", fontsize=6)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"panel_{spec}.{ext}", dpi=400, bbox_inches="tight")
        plt.close(fig); print(f"[panels] {spec} -> {out}")


if __name__ == "__main__":
    main()
