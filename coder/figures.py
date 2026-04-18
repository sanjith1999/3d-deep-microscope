"""
Figures for one coder-design run: training curves, phase masks, illumination planes and depth correlation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from coder.losses import plane_matrix, coherence_profile


def training_curves(history: dict, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    s = history["step"]
    for k in ("total", "decorr", "sym", "pdf"):
        ax[0].plot(s, history[k], label=k)
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
    ax[0].legend(); ax[0].set_title("Surrogate objective")
    ax[1].plot(s, history["coherence"], "C3", label="raw (adjacent-limited)")
    if "coherence_res" in history:
        ax[1].plot(s, history["coherence_res"], "C0", label="beyond target resolution")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("max off-diag |corr|")
    ax[1].legend(); ax[1].set_title("Axial coherence (lower=better)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def phase_masks(phases: torch.Tensor, out: Path) -> None:
    p = phases.detach().cpu().numpy()
    n = p.shape[0]
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    ax = np.atleast_1d(ax)
    for i in range(n):
        im = ax[i].imshow(np.angle(np.exp(1j * p[i])), cmap="twilight")
        ax[i].set_title(f"layer {i}"); ax[i].axis("off")
        fig.colorbar(im, ax=ax[i], fraction=0.046)
    fig.suptitle("Designed diffractive phase masks (wrapped)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def speckle_planes(vol: torch.Tensor, readout_z, out_png: Path, out_gif: Path) -> None:
    """vol: (T,K,N,N), averaged over patterns for a clean per-depth view."""
    v = vol.mean(dim=0).detach().cpu().numpy()      # (K, N, N)
    K = v.shape[0]
    vmax = float(v.max())
    cols = min(K, 8); rows = int(np.ceil(K / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    ax = np.atleast_1d(ax).ravel()
    for k in range(K):
        ax[k].imshow(v[k], cmap="inferno", vmin=0, vmax=vmax)
        ax[k].set_title(f"z={readout_z[k]:+.1f} um", fontsize=8); ax[k].axis("off")
    for k in range(K, len(ax)):
        ax[k].axis("off")
    fig.suptitle("Excitation intensity vs depth (pattern-averaged)")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)

    frames = [(255 * np.clip(v[k] / (vmax + 1e-12), 0, 1)).astype(np.uint8) for k in range(K)]
    imageio.mimsave(out_gif, frames, duration=0.2)


def axial_correlation(vol: torch.Tensor, dz_um: float, out: Path) -> None:
    with torch.no_grad():
        G = (plane_matrix(vol) @ plane_matrix(vol).t()).abs().cpu().numpy()
        seps, mean_c, max_c = coherence_profile(vol)
    sep_um = seps.cpu().numpy() * dz_um

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    im = ax[0].imshow(G, cmap="viridis", vmin=0, vmax=1)
    ax[0].set_title("Plane-to-plane |correlation|\n(anti-diagonal = +d/-d symmetry)")
    ax[0].set_xlabel("depth plane"); ax[0].set_ylabel("depth plane")
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    ax[1].plot(sep_um, max_c.cpu().numpy(), "o-", label="max")
    ax[1].plot(sep_um, mean_c.cpu().numpy(), "s-", label="mean")
    ax[1].set_xlabel("axial separation (um)"); ax[1].set_ylabel("|correlation|")
    ax[1].set_ylim(0, 1); ax[1].legend()
    ax[1].set_title("Axial decorrelation vs separation")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def symmetry_comparison(scores: dict, dz_um: float, out: Path) -> None:
    names = list(scores.keys())
    n = len(names)

    fig, ax = plt.subplots(1, n + 1, figsize=(4.0 * (n + 1), 4.0))
    im = None
    for i, name in enumerate(names):
        s = scores[name]
        im = ax[i].imshow(s["gram"], cmap="viridis", vmin=0, vmax=1)
        ax[i].set_title(f"{name}\nsym(+d/-d)={s['sym']:.3f}  coh@{s['res_sep']}={s['coherence_res']:.3f}",
                        fontsize=10)
        ax[i].set_xlabel("depth plane")
        if i == 0:
            ax[i].set_ylabel("depth plane")
    fig.colorbar(im, ax=ax[n - 1], fraction=0.046)

    for name in names:
        s = scores[name]
        ax[n].plot(s["seps"] * dz_um, s["max_c"], "o-", ms=3, label=name)
    ax[n].set_xlabel("axial separation (um)"); ax[n].set_ylabel("max |correlation|")
    ax[n].set_ylim(0, 1.02); ax[n].legend(); ax[n].set_title("Axial decorrelation vs separation")
    fig.suptitle("Illumination decorrelation: amplitude (DMD) vs diffractive coding", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def all_figures(result: dict) -> None:
    out = result["out_dir"]
    m = result["manifest"]; cfg = m["config"]
    nz, dz, focus = cfg["nz"], cfg["dz_um"], m["focus_idx"]
    z_um = [(i - focus) * dz for i in range(nz)]
    training_curves(result["history"], out / "training_curves.png")
    phase_masks(torch.stack(result["model"].phases), out / "phase_masks.png")
    speckle_planes(result["volume"], z_um, out / "speckle_planes.png", out / "speckle_stack.gif")
    axial_correlation(result["volume"], dz, out / "axial_correlation.png")
    print(f"figures written to {out}")
