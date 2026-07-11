#!/usr/bin/env python3
"""
Structure metrics for every saved reconstruction: 3D SSIM, NRMSE, bead localisation, spectral fidelity, FSC.
"""
from __future__ import annotations
import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from pipeline.metrics_paper import nrmse, ssim3d, ssim3d_foreground, bead_localisation, fsc_directional, fidelity_directional
from pipeline.specimens import BEADS

PAT = re.compile(r"^(?P<op>.+)_k(?P<k>\d+)_(?P<h>[0-9a-f]{10})_(?P<spec>.+?)_s(?P<seed>-?\d+)_q(?P<q>[\d.]+)$")


def main() -> int:
    ap = argparse.ArgumentParser(description="Structure-aware metrics for saved reconstructions.")
    ap.add_argument("--dir", default="results/recon/compression")
    ap.add_argument("--redo", action="store_true", help="rescore every volume (rewrite volume_metrics.csv)")
    a = ap.parse_args()
    d = _ROOT / a.dir
    gts = {k: v.numpy() for k, v in torch.load(_ROOT / "results/recon/ops/specimens.pt")["gt"].items()}
    out = d / "volume_metrics.csv"
    done = set()
    if a.redo and out.exists():
        out.unlink()
    if out.exists():
        with open(out, newline="") as f:
            done = {r["file"] for r in csv.DictReader(f)}
    groups = defaultdict(dict); rows = []
    for f in sorted((d / "volumes").glob("*.npz")):
        m = PAT.match(f.stem)
        if not m:
            continue
        est = np.load(f)["est"].astype(np.float64); gt = gts[m["spec"]]
        key = (m["op"], m["k"], m["h"], m["spec"], m["q"])
        groups[key][int(m["seed"])] = est
        if f.name in done:
            continue
        r = {"file": f.name, "op": m["op"], "k": int(m["k"]), "subset": m["h"], "specimen": m["spec"],
             "seed": int(m["seed"]), "q": float(m["q"]), "ssim3d": ssim3d(gt, est),
             "ssim3d_fg": ssim3d_foreground(gt, est), "nrmse": nrmse(gt, est)}
        g = fidelity_directional(gt, est)
        r["fidelity_res_lateral_um"] = g["resolution_lateral_um"]; r["fidelity_res_axial_um"] = g["resolution_axial_um"]
        if m["spec"] == "beads":
            r.update({f"bead_{k}": v for k, v in bead_localisation(est, BEADS, gt=gt).items()})
        rows.append(r)
    if rows:
        keys = sorted({k for r in rows for k in r}); new = not out.exists()
        if not new:
            with open(out, newline="") as f:
                keys = next(csv.reader(f))
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
    pairs = []
    for (op, k, h, spec, q), by_seed in groups.items():
        s = sorted(x for x in by_seed if x >= 0)
        for i in range(len(s) - 1):
            g = fsc_directional(by_seed[s[i]], by_seed[s[i + 1]])
            pairs.append({"op": op, "k": int(k), "subset": h, "specimen": spec, "q": float(q), "seeds": f"{s[i]}-{s[i + 1]}",
                          "res_lateral_um": g["resolution_lateral_um"], "res_axial_um": g["resolution_axial_um"]})
    if pairs:
        with open(d / "fsc_pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    print(f"[analyse] {len(rows)} new volumes scored -> {out}; {len(pairs)} seed pairs -> {d / 'fsc_pairs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
