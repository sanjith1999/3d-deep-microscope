#!/usr/bin/env python3
"""
Figures for the point-emitter information analysis.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = _ROOT / "results/paper/information"
LAB = {"dmd": "DMD (in-focus amplitude)", "random1": "1-layer phase plate", "random2": "2-layer diffractive element"}
COL = {"dmd": "#7a7a7a", "random1": "#d07a1f", "random2": "#1f6fb4"}


def main() -> int:
    r = json.loads((D / "point_information.json").read_text(encoding="utf-8"))
    dz, fz = r["dz_um"], r["focus_plane"]
    coders = [c for c in ("dmd", "random1", "random2") if any(x["coder"] == c for x in r["results"])]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for c in coders:
        x = next(x for x in r["results"] if x["coder"] == c and x["rule"] == "all" and x["dose"] == "frame")
        zs = [(i - fz) * dz for i in range(len(x["crb_z_by_plane"]))]
        ok = lambda i: 1 <= i <= 2 * fz - 1                    # both the plane and its mirror interior
        cz = [(z, v) for i, (z, v) in enumerate(zip(zs, x["crb_z_by_plane"])) if v is not None and ok(i)]
        ax[0].semilogy(*zip(*cz), "o-", ms=3, color=COL[c], label=LAB[c])
        dm = [(z, max(v, 1e-3)) for i, (z, v) in enumerate(zip(zs, x["dprime_by_plane"])) if v is not None and ok(i)]
        ax[1].semilogy(*zip(*dm), "o-", ms=3, color=COL[c], label=LAB[c])
    ax[0].set(xlabel="depth from excitation focus (µm)", ylabel="axial CRB σ_z (µm)",
              title="Axial localisation bound, 256 frames")
    ax[1].axhline(1, ls=":", c="k", lw=0.8)
    ax[1].set(xlabel="depth from excitation focus (µm)", ylabel="mirror discriminability d′ (σ)",
              title="Separation of +d and −d (1 = noise level)")
    ax[0].legend(fontsize=7, frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(D / f"fig_information_depth.{ext}", dpi=200)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ks = sorted({x["k"] for x in r["results"]})
    for c in coders:
        for dose, ls in (("frame", "--"), ("total", "-")):
            pz, pd = [], []
            for k in ks:
                xs = [x for x in r["results"] if x["coder"] == c and x["k"] == k and x["dose"] == dose
                      and x["rule"] in ("random", "all")]
                if xs:
                    pz.append((k, st.median(x["crb_z_um"]["median"] for x in xs)))
                    pd.append((k, max(st.median(x["dprime_mirror"]["median"] for x in xs), 1e-3)))
            ax[0].loglog(*zip(*pz), ls, marker="o", ms=3, color=COL[c],
                         label=f"{LAB[c]}, {'equal total' if dose == 'total' else 'equal per-frame'} dose")
            ax[1].loglog(*zip(*pd), ls, marker="o", ms=3, color=COL[c])
    for a, yl in zip(ax, ("median axial CRB σ_z (µm)", "median mirror discriminability d′")):
        a.set(xlabel="frames k (of 256)", ylabel=yl); a.set_xticks(ks); a.set_xticklabels(ks); a.grid(alpha=0.3)
    ax[1].axhline(1, ls=":", c="k", lw=0.8)
    ax[0].legend(fontsize=6, frameon=False)
    ax[0].set_title("Axial information vs number of frames"); ax[1].set_title("±d separation vs number of frames")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(D / f"fig_information_frames.{ext}", dpi=200)
    print(f"[plot] -> {D}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
