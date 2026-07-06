#!/usr/bin/env python3
"""
Operator conditioning from sampled columns at every depth and their lateral neighbours.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-300)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sampled-column conditioning of registered operators.")
    ap.add_argument("--ops", required=True)
    ap.add_argument("--lat_step", type=int, default=4)
    ap.add_argument("--out", default="results/paper/conditioning.json")
    a = ap.parse_args()
    ops_dir = _ROOT / "results/recon/ops"; res = {"definition": __doc__.split("    python")[0].strip(), "ops": {}}
    outp = _ROOT / a.out
    if outp.exists():
        res["ops"] = json.loads(outp.read_text(encoding="utf-8")).get("ops", {})
    for op in a.ops.split(","):
        meta = json.loads((ops_dir / op / "meta.json").read_text())
        rz, rx, ry = meta["recon_shape"]
        xs = list(range(a.lat_step // 2, rx - 1, a.lat_step)); lat = [(x, y) for x in xs for y in xs]
        cid = lambda z, x, y: z * rx * ry + x * ry + y
        idx = [cid(z, x, y) for z in range(rz) for (x, y) in lat]
        idxl = [cid(z, x + 1, y) for z in range(rz) for (x, y) in lat]
        t = time.time()
        A = torch.load(_ROOT / meta["A"], map_location="cpu", mmap=True)
        C = A[:, idx].double().numpy().reshape(A.shape[0], rz, len(lat))
        L = A[:, idxl].double().numpy().reshape(A.shape[0], rz, len(lat))
        ca = np.concatenate([corr(C[:, z], C[:, z + 1]) for z in range(rz - 1)])
        cl = np.concatenate([corr(C[:, z], L[:, z]) for z in range(rz)])
        cm = np.concatenate([corr(C[:, z], C[:, 14 - z]) for z in range(rz) if 0 <= 14 - z < rz and 14 - z != z])
        S = C.reshape(A.shape[0], -1); S = S / np.linalg.norm(S, axis=0, keepdims=True)
        s = np.linalg.svd(S, compute_uv=False)
        p = s ** 2 / (s ** 2).sum()
        r = {"T": meta["meas_shape"][0], "n_columns": S.shape[1], "rows": S.shape[0],
             "corr_axial_mean": float(ca.mean()), "corr_lateral_mean": float(cl.mean()), "corr_mirror_mean": float(cm.mean()),
             "sigma_min": float(s.min()), "sigma_max": float(s.max()), "cond": float(s.max() / max(s.min(), 1e-300)),
             "erank_participation": float((s ** 2).sum() ** 2 / (s ** 4).sum()),
             "erank_entropy": float(np.exp(-(p * np.log(p + 1e-300)).sum())), "light_kept": meta.get("light_kept")}
        res["ops"][op] = r
        print(f"[cond] {op:20s} T={r['T']:3d} corr axial {r['corr_axial_mean']:.3f} lateral {r['corr_lateral_mean']:.3f} "
              f"mirror {r['corr_mirror_mean']:.3f} | sigma_min {r['sigma_min']:.4f} cond {r['cond']:.1f} "
              f"erank {r['erank_participation']:.1f}/{r['erank_entropy']:.1f} of {r['n_columns']} ({time.time() - t:.0f}s)", flush=True)
    outp.parent.mkdir(parents=True, exist_ok=True); outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[cond] -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
