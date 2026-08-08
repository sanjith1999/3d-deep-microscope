#!/usr/bin/env python3
"""
Excitation volumes and camera frames of the one-plate device and the DMD for the acquisition figure.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon"):
    sys.path.insert(0, str(p))
import numpy as np
import torch
from configs.default import get_config
from operator_assembly import illumination


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--idx", default="0,136"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); idx = [int(i) for i in a.idx.split(",")]
    cfg = get_config(); cfg.device = a.device
    out = _ROOT / "results/paper/acquisition"; out.mkdir(parents=True, exist_ok=True)
    arrays, man = {"pattern_idx": np.array(idx)}, {"pattern_idx": idx, "grid": "analysis 32 x 96 x 96 at 0.32 um (z, x, y)",
           "normalisation": "matched_dose over all 256 patterns of each operator: mean intensity 30 (same as the operators)",
           "frames": "clean registered measurements Y (fine object), neuronal subvolume 0, 12 x 12 detector (2.56 um pixels)",
           "focus_plane": None, "coders": {}}
    from coder.diffractive import _angle_grid
    al, th = _angle_grid(cfg.n_alphas, cfg.n_thetas)
    for name, op in (("plate", "random1L_gap66"), ("dmd", "dmd_fast")):
        mp = _ROOT / "results/recon/ops" / op / "meta.json"
        if not mp.exists():
            print(f"[acq] {op} not registered; skipped"); continue
        meta = json.loads(mp.read_text())
        H2 = illumination(meta, cfg, a.device).cpu().numpy()                   # (256, 32, 96, 96)
        Y = torch.load(_ROOT / "results/recon/ops" / op / "Y.pt")["neural_cell"].reshape(256, 12, 12).numpy()
        arrays[f"H2_{name}"] = H2[idx].astype(np.float32)
        arrays[f"xz_{name}"] = H2[idx][:, :, :, 48].astype(np.float32)
        arrays[f"axial_mean_{name}"] = H2.mean((0, 2, 3)).astype(np.float32)
        arrays[f"frames_{name}"] = Y[idx].astype(np.float32)
        man["coders"][name] = {"operator": op, "coder": meta["coder"], "layers": meta.get("physics", {}).get("n_layers"),
                               "paraxial": meta.get("physics", {}).get("d2nn_paraxial", False),
                               "directions_alpha_theta_rad": [[float(al[i // cfg.n_thetas]), float(th[i % cfg.n_thetas])] for i in idx]
                               if meta["coder"] != "dmd" else "DMD: binary patterns, no tilt geometry"}
        del H2
    man["focus_plane"] = cfg.nz // 2 - 1
    np.savez_compressed(out / "acquisition_arrays.npz", **arrays)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in ("plate", "dmd") if f"H2_{n}" in arrays]; f = man["focus_plane"]; P = 0.32
    fig, ax = plt.subplots(len(names) * 2, 3, figsize=(6.4, 2.2 * len(names) * 2), gridspec_kw={"width_ratios": [1, 1, 0.6]})
    for r, n in enumerate(names):
        for j in range(2):
            row = 2 * r + j; H = arrays[f"H2_{n}"][j]; v = float(H.max())
            ax[row, 0].imshow(H[f], cmap="magma", vmin=0, vmax=v, extent=[0, 96 * P, 96 * P, 0])
            ax[row, 1].imshow(arrays[f"xz_{n}"][j], cmap="magma", vmin=0, vmax=v, extent=[0, 96 * P, 32 * P, 0], aspect="equal")
            ax[row, 1].axhline((f + 0.5) * P, color="c", lw=0.4, ls=":")
            ax[row, 2].imshow(arrays[f"frames_{n}"][j], cmap="gray", extent=[0, 12 * 2.56, 12 * 2.56, 0])
            ax[row, 0].set_ylabel(f"{n}, pattern {idx[j]}", fontsize=6)
            for c in range(3):
                ax[row, c].tick_params(labelsize=5)
    ax[0, 0].set_title("excitation, xy at focus", fontsize=7); ax[0, 1].set_title("excitation, xz slice (y = centre)", fontsize=7)
    ax[0, 2].set_title("camera frame (neuronal vol.)", fontsize=7)
    fig.suptitle("Verified simulated acquisition data (draft layout; axes in µm; per-row shared limits)", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"preview.{ext}", dpi=300, bbox_inches="tight")
    print(f"[acq] -> {out}")


if __name__ == "__main__":
    main()
