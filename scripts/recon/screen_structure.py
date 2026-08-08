#!/usr/bin/env python3
"""
Pick a structure-rich neuronal subvolume for the qualitative figure from reference volumes only.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import torch
import torch.nn.functional as F
from pipeline.specimens import STORED


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/paper/structure_screen"); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(); out = _ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    X = torch.load(_ROOT / STORED["neural_cell"], map_location="cpu")
    rows = []
    for i in range(X.shape[0]):
        x = X[i].float().reshape(32, 128, 128)
        x = F.adaptive_avg_pool3d(x[None, None], (32, 96, 96))[0, 0]; x = x / (x.max() + 1e-12)      # as stored_specimen
        occ = float((x > 0.1).float().mean())
        mip, arg = x.max(0); fg = mip > 0.2
        area = float(fg.float().mean()); spread = float(arg[fg].float().std()) if fg.sum() > 1 else 0.0
        rows.append({"index": i, "occupancy_pct": 100 * occ, "mip_area_pct": 100 * area, "depth_spread_planes": spread,
                     "eligible": 0.01 <= occ <= 0.08, "score": area * spread if 0.01 <= occ <= 0.08 else 0.0})
    rows.sort(key=lambda r: -r["score"])
    with open(out / "ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = rows[:a.top]
    fig, ax = plt.subplots(2, len(top), figsize=(1.6 * len(top), 2.6), gridspec_kw={"height_ratios": [3, 1]})
    for j, r in enumerate(top):
        x = X[r["index"]].float().reshape(32, 128, 128)
        x = F.adaptive_avg_pool3d(x[None, None], (32, 96, 96))[0, 0]; x = (x / x.max()).numpy()
        mip = x.max(0); dep = x.argmax(0) / 31.0
        rgb = plt.cm.viridis(dep)[..., :3] * mip[..., None]
        ax[0, j].imshow(rgb); ax[1, j].imshow(x.max(1), cmap="magma", aspect="equal")
        ax[0, j].set_title(f"p{r['index']}\nocc {r['occupancy_pct']:.1f}%  spread {r['depth_spread_planes']:.1f}", fontsize=5)
        for rr in range(2):
            ax[rr, j].set_xticks([]); ax[rr, j].set_yticks([])
    fig.suptitle("REFERENCE volumes only (no reconstruction): top structure scores; xy MIP coloured by depth, xz MIP", fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"top_references.{ext}", dpi=300, bbox_inches="tight")
    print("rank index occ% area% spread")
    for r in rows[:a.top]:
        print(f"{r['index']:5d} {r['occupancy_pct']:5.2f} {r['mip_area_pct']:5.2f} {r['depth_spread_planes']:5.2f} {r['score']:.4f}")
    print(f"eligible {sum(r['eligible'] for r in rows)} / {len(rows)}; crop 0 rank {[r['index'] for r in rows].index(0) + 1}")


if __name__ == "__main__":
    main()
