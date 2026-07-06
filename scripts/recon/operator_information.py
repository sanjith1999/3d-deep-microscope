#!/usr/bin/env python3
"""
Decoder-free information of an operator: Cramer-Rao bounds and mirror discriminability per voxel.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from pipeline.specimens import mirror_plane

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PITCH = 0.64          # um, reconstruction grid (lateral and axial)


def load_parts(paths):
    """Rows of a partly built operator: the saved build parts concatenated in pattern order."""
    return torch.cat([torch.load(_ROOT / p, map_location="cpu", mmap=True) for p in paths], 0)


def load_columns(meta, ops_dir, lat_step, A=None):
    rz, rx, ry = meta["recon_shape"]
    xs = list(range(lat_step // 2, rx, lat_step)); ys = list(range(lat_step // 2, ry, lat_step))
    lat = [(x, y) for x in xs for y in ys]
    idx = [z * rx * ry + x * ry + y for z in range(rz) for (x, y) in lat]
    A = torch.load(_ROOT / meta["A"], map_location="cpu", mmap=True) if A is None else A
    t = time.time()
    C = A[:, idx].double().numpy().reshape(A.shape[0], rz, len(lat))
    print(f"[info] {meta['name']}: {C.shape[2]} lateral x {rz} depths columns read in {time.time() - t:.0f}s", flush=True)
    # neighbours for the lateral derivative (x +- 1)
    idxp = [z * rx * ry + min(x + 1, rx - 1) * ry + y for z in range(rz) for (x, y) in lat]
    idxm = [z * rx * ry + max(x - 1, 0) * ry + y for z in range(rz) for (x, y) in lat]
    Cxp = A[:, idxp].double().numpy().reshape(A.shape[0], rz, len(lat))
    Cxm = A[:, idxm].double().numpy().reshape(A.shape[0], rz, len(lat))
    return C, Cxp, Cxm, lat


def fisher(a, d):
    aa = (a * a).sum(0); dd = (d * d).sum(0); ad = (a * d).sum(0)
    return np.maximum(dd - ad ** 2 / np.maximum(aa, 1e-300), 0.0)


def analyse(C, Cxp, Cxm, rows, sigma):
    """Per (z, lateral): crb_z, crb_x (um) and d'_mirror, for the given measurement rows."""
    a = C[rows]; rz = a.shape[1]
    crb_z = np.full((rz, a.shape[2]), np.nan); dz = np.full_like(crb_z, np.nan)
    for z in range(1, rz - 1):
        d = (a[:, z + 1] - a[:, z - 1]) / (2 * PITCH)
        I = fisher(a[:, z], d) / sigma ** 2
        crb_z[z] = 1 / np.sqrt(np.maximum(I, 1e-300))
    dx = (Cxp[rows] - Cxm[rows]) / (2 * PITCH)
    crb_x = 1 / np.sqrt(np.maximum(fisher(a, dx) / sigma ** 2, 1e-300))
    for z in range(rz):
        zm = mirror_plane(z)
        if 0 <= zm < rz and zm != z:
            u, v = a[:, z], a[:, zm]
            c = (u * v).sum(0) / np.maximum((v * v).sum(0), 1e-300)
            dz[z] = np.linalg.norm(u - c * v, axis=0) / sigma
    return crb_z, crb_x, dz


def summarise(x):
    v = x[np.isfinite(x)]
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max())} if v.size else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Cramer-Rao and mirror-discriminability analysis of operators.")
    ap.add_argument("--ops", default="random_gap66")
    ap.add_argument("--rules", default="first,random,spread", help="subset rules from subsets.json (k < T)")
    ap.add_argument("--ks", default="8,16,32,64,128,256")
    ap.add_argument("--subsets", default="results/recon/compression/subsets.json")
    ap.add_argument("--q", type=float, default=25.0)
    ap.add_argument("--ref", default="blood_cell", help="specimen fixing sigma (tuning specimen; never a test one)")
    ap.add_argument("--lat_step", type=int, default=4, help="lateral sampling of voxels (4 -> 12 x 12 positions)")
    ap.add_argument("--t_ref", type=int, default=256, help="frames of the full scan (dose 'total')")
    ap.add_argument("--parts", default="", help="analyse saved build parts instead of a registered operator "
                    "(comma list, pattern order); needs --like and --label")
    ap.add_argument("--like", default="random_gap66", help="--parts: registered operator with the same geometry; "
                    "its meta gives the grid and its Y fixes sigma (same dose normalisation)")
    ap.add_argument("--label", default="partial", help="--parts: name for the analysed operator")
    ap.add_argument("--out", default="results/paper/information")
    a = ap.parse_args()
    ops_dir = _ROOT / "results/recon/ops"; out = _ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    subs = json.loads((_ROOT / a.subsets).read_text()) if (_ROOT / a.subsets).exists() else {}
    report = {"definition": __doc__.split("    python")[0].strip(), "q": a.q, "ref": a.ref, "lat_step": a.lat_step,
              "pitch_um": PITCH, "results": []}
    for op in (a.ops.split(",") if not a.parts else [a.label]):
        Apart = None
        if a.parts:
            Apart = load_parts(a.parts.split(","))
            meta = dict(json.loads((ops_dir / a.like / "meta.json").read_text()))
            nd1, nd2 = meta["meas_shape"][1:]
            meta.update(name=a.label, coder=a.label, meas_shape=[Apart.shape[0] // (nd1 * nd2), nd1, nd2])
            yref = torch.load(ops_dir / a.like / "Y.pt")[a.ref].numpy()
        else:
            meta = json.loads((ops_dir / op / "meta.json").read_text())
            yref = torch.load(ops_dir / op / "Y.pt")[a.ref].numpy()
        T, nd1, nd2 = meta["meas_shape"]; rp = nd1 * nd2
        sigma0 = float(yref.mean()) * 10 ** (-a.q / 10)
        C, Cxp, Cxm, lat = load_columns(meta, ops_dir, a.lat_step, Apart)
        sets = []
        for k in [int(x) for x in a.ks.split(",")]:
            if a.parts:
                if k <= T:
                    sets.append(("first", 0, k, list(range(k))))
                continue
            if k >= T:
                sets.append(("all", 0, T, list(range(T))))
            else:
                for rule in a.rules.split(","):
                    for draw, pats in enumerate(subs.get(op, {}).get(str(k), {}).get(rule, [])):
                        sets.append((rule, draw, k, pats))
        for (rule, draw, k, pats) in sets:
            rows = np.concatenate([np.arange(p * rp, (p + 1) * rp) for p in sorted(pats)])
            for dose in ("frame", "total"):
                sigma = sigma0 * (math.sqrt(k / a.t_ref) if dose == "total" else 1.0)
                cz, cx, dm = analyse(C, Cxp, Cxm, rows, sigma)
                rec = {"op": op, "coder": meta["coder"], "rule": rule, "draw": draw, "k": k, "M": k * rp, "dose": dose,
                       "sigma": sigma, "crb_z_um": summarise(cz), "crb_x_um": summarise(cx), "dprime_mirror": summarise(dm),
                       "crb_z_by_depth_median": [float(np.nanmedian(cz[z])) if np.isfinite(cz[z]).any() else None for z in range(cz.shape[0])],
                       "dprime_mirror_by_depth_median": [float(np.nanmedian(dm[z])) if np.isfinite(dm[z]).any() else None for z in range(dm.shape[0])]}
                report["results"].append(rec)
                if dose == "frame":
                    np.savez_compressed(out / f"{op}_{rule}{draw}_k{k}.npz", crb_z=cz, crb_x=cx, dprime_mirror=dm,
                                        lateral=np.array(lat), patterns=np.array(sorted(pats)))
                print(f"[info] {op:18s} k={k:3d} {rule:6s}#{draw} {dose:5s}  sigma_z median {rec['crb_z_um']['median']:.4f} um "
                      f"(p90 {rec['crb_z_um']['p90']:.4f})  sigma_x {rec['crb_x_um']['median']:.4f} um  "
                      f"d'_mirror median {rec['dprime_mirror']['median']:.1f} (p10 {rec['dprime_mirror']['p10']:.1f})", flush=True)
        (out / f"information_{(a.label if a.parts else a.ops).replace(',', '+')}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[info] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
