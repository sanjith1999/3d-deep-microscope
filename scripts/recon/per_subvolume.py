#!/usr/bin/env python3
"""
Per-subvolume paired summary of depth error, SSIM and PSNR against each subvolume's full scan.
"""
from __future__ import annotations
import argparse, csv, json, statistics as st, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="16,32,64,256"); a = ap.parse_args()
    d = _ROOT / a.dir; out = _ROOT / a.out; out.mkdir(parents=True, exist_ok=True); ks = [int(k) for k in a.ks.split(",")]
    runs = [json.loads(l) for f in sorted(d.glob("runs*.jsonl")) for l in open(f) if l.strip()]
    vm = list(csv.DictReader(open(d / "volume_metrics.csv")))
    gts = torch.load(_ROOT / "results/recon/ops/specimens.pt")["gt"]
    occ = {s: float((g > 0.1 * g.max()).float().mean()) for s, g in gts.items() if s.startswith("neural_cell")}
    specs = sorted({r["specimen"] for r in runs if r["specimen"].startswith("neural_cell")}, key=lambda s: (s != "neural_cell", s))
    rows = []
    for s in specs:
        for seed in sorted({r["seed"] for r in runs if r["specimen"] == s}):
            def R(k):  return [r for r in runs if r["specimen"] == s and r["seed"] == seed and r["k"] == k]
            def V(k):  return [v for v in vm if v["specimen"] == s and int(v["seed"]) == seed and int(v["k"]) == k]
            ref = R(256); refv = V(256)
            if not ref:
                continue
            r_dep = ref[0]["local_depth_err"] * 0.64; r_psnr = ref[0]["psnr"]; r_fg = float(refv[0]["ssim3d_fg"]) if refv else None
            for k in ks:
                rr, vv = R(k), V(k)
                if not rr:
                    continue
                dep = [r["local_depth_err"] * 0.64 for r in rr]; ps = [r["psnr"] for r in rr]; fg = [float(v["ssim3d_fg"]) for v in vv]
                sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
                rows.append({"subvolume": s, "occupancy_pct": round(100 * occ.get(s, float("nan")), 3), "noise_seed": seed, "k": k,
                             "compression": 256 / k, "n_draws": len(rr),
                             "depth_err_um_mean": st.mean(dep), "depth_err_um_sd": sd(dep), "d_depth_um_vs_own_full": st.mean(dep) - r_dep,
                             "fg_ssim_mean": st.mean(fg) if fg else "", "d_fg_ssim_vs_own_full": (st.mean(fg) - r_fg) if fg and r_fg is not None else "",
                             "ssim3d_mean": st.mean(float(v["ssim3d"]) for v in vv) if vv else "",
                             "psnr_mean": st.mean(ps), "psnr_sd": sd(ps), "d_psnr_vs_own_full": st.mean(ps) - r_psnr})
    with open(out / "per_subvolume_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "per_subvolume_paired.md", "w", encoding="utf-8") as f:
        f.write("| subvolume | occupancy % | seed | k | depth err (µm) | Δdepth vs own full (µm) | fg SSIM | Δfg SSIM | PSNR (dB) | ΔPSNR |\n|" + "---|" * 10 + "\n")
        for r in rows:
            g = lambda x, n=3: f"{x:.{n}f}" if isinstance(x, float) else str(x)
            f.write(f"| {r['subvolume']} | {r['occupancy_pct']} | {r['noise_seed']} | {r['k']} | {g(r['depth_err_um_mean'])} ± {g(r['depth_err_um_sd'])} | "
                    f"{g(r['d_depth_um_vs_own_full'])} | {g(r['fg_ssim_mean'])} | {g(r['d_fg_ssim_vs_own_full'])} | {g(r['psnr_mean'], 2)} | {g(r['d_psnr_vs_own_full'], 2)} |\n")
    print(f"[per_subvolume] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
