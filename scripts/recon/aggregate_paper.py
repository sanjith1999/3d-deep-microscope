#!/usr/bin/env python3
"""
Manuscript tables from the compression, noise and cohort runs (test data only).
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PITCH = 0.64


def group_of(spec):
    return "cohort" if spec.startswith("neural_cell_p") else spec


def load_runs(d):
    out = []
    for f in sorted(d.glob("runs*.jsonl")):
        for l in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    return out


def load_volumes(d):
    p = d / "volume_metrics.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {(r["op"], int(r["k"]), r["subset"], r["specimen"], int(r["seed"])): r for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate test-only results into the paper tables.")
    ap.add_argument("--dirs", required=True, help="comma list of dir:q")
    ap.add_argument("--ops", default="random_gap66")
    ap.add_argument("--out", default="results/paper/tables")
    a = ap.parse_args()
    out = _ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    ops = set(a.ops.split(","))
    import hashlib
    rows = defaultdict(list)
    for entry in a.dirs.split(","):
        dname, q = entry.rsplit(":", 1); q = float(q); d = _ROOT / dname
        if not d.exists():
            continue
        vols = load_volumes(d)
        for r in load_runs(d):
            if r["op"] not in ops or r["seed"] not in (0, 1, 2) or r["rule"] not in ("random", "all"):
                continue
            g = group_of(r["specimen"])
            rec = {"psnr": r["psnr"], "depth_um": r["local_depth_err"] * PITCH}
            if "ghost" in r:
                rec["ghost"] = r["ghost"]
            # join the structure metrics of the same volume (subset hash as in compression_subsets.save_volume)
            subs = json.loads((d / "subsets.json").read_text()).get(r["op"], {}).get(str(r["k"]), {})
            pats = subs.get(r["rule"], [[]])[r["draw"]] if subs.get(r["rule"]) else None
            if pats is not None:
                h = hashlib.sha1(json.dumps(sorted(int(p) for p in pats)).encode()).hexdigest()[:10]
                v = vols.get((r["op"], r["k"], h, r["specimen"], r["seed"]))
                if v:
                    for c, name in (("ssim3d", "ssim"), ("ssim3d_fg", "ssim_fg"), ("fidelity_res_axial_um", "res_axial_um"),
                                    ("fidelity_res_lateral_um", "res_lateral_um"), ("bead_jaccard", "jaccard"),
                                    ("bead_rms_axial_um", "bead_axial_rms_um")):
                        if v.get(c) not in (None, "", "nan", "inf"):
                            rec[name] = float(v[c])
            rec["_spec"] = r["specimen"]; rec["_seed"] = r["seed"]
            rows[(q, g, r["k"])].append(rec)
    # paired loss relative to the SAME object's full scan (k = 256, same noise seed when available)
    full = {}
    for (q, g, k), v in rows.items():
        if k == 256:
            for x in v:
                full.setdefault((q, x["_spec"]), {})[x["_seed"]] = x
    for (q, g, k), v in rows.items():
        for x in v:
            ref = full.get((q, x["_spec"]), {})
            f = ref.get(x["_seed"]) or (next(iter(ref.values())) if ref else None)
            if f is not None:
                x["dpsnr"] = x["psnr"] - f["psnr"]; x["ddepth_um"] = x["depth_um"] - f["depth_um"]
    metrics = ["psnr", "dpsnr", "depth_um", "ddepth_um", "ssim", "ssim_fg", "res_axial_um", "res_lateral_um", "ghost", "jaccard",
               "bead_axial_rms_um"]
    table = []
    for (q, g, k), v in sorted(rows.items()):
        t = {"q": q, "photons_per_px_frame": 10 ** (q / 5), "specimen": g, "k": k, "measurements": 144 * k, "n": len(v)}
        for m in metrics:
            xs = [x[m] for x in v if m in x]
            if xs:
                t[f"{m}_mean"] = float(np.mean(xs)); t[f"{m}_std"] = float(np.std(xs)); t[f"{m}_n"] = len(xs)
        table.append(t)
    keys = sorted({k for t in table for k in t}, key=lambda s: (s not in ("q", "photons_per_px_frame", "specimen", "k", "measurements", "n"), s))
    with open(out / "compression_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(table)
    md = ["| q (photons/px/frame) | specimen | k | n | depth error (µm) | Δdepth vs own full scan (µm) | axial res. (µm) | "
          "lateral res. (µm) | foreground SSIM | SSIM | PSNR (dB) | ΔPSNR vs own full scan | ghost | Jaccard |",
          "|" + "---|" * 14]
    fmt = lambda t, m, p=2: f"{t[m + '_mean']:.{p}f} ± {t[m + '_std']:.{p}f}" if m + "_mean" in t else ""
    for t in table:
        md.append(f"| {t['q']:g} ({t['photons_per_px_frame']:.0f}) | {t['specimen']} | {t['k']} | {t['n']} | "
                  f"{fmt(t, 'depth_um', 3)} | {fmt(t, 'ddepth_um', 3)} | {fmt(t, 'res_axial_um')} | {fmt(t, 'res_lateral_um')} | "
                  f"{fmt(t, 'ssim_fg', 3)} | {fmt(t, 'ssim', 3)} | {fmt(t, 'psnr')} | {fmt(t, 'dpsnr')} | "
                  f"{fmt(t, 'ghost', 3)} | {fmt(t, 'jaccard')} |")
    (out / "compression_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("depth_um", "local depth error (µm)"), ("res_axial_um", "axial resolution (µm)"),
              ("ssim_fg", "foreground SSIM"), ("dpsnr", "ΔPSNR vs own full scan (dB)")]
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for (q, g) in sorted({(t["q"], t["specimen"]) for t in table}):
        if g == "beads":
            continue
        ts = sorted([t for t in table if t["q"] == q and t["specimen"] == g], key=lambda t: t["k"])
        for a_, (m, lab) in zip(ax, panels):
            pts = [(t["k"], t[m + "_mean"], t[m + "_std"]) for t in ts if m + "_mean" in t]
            if pts:
                a_.errorbar(*zip(*pts), marker="o", ms=3, capsize=2, label=f"{g}, {10 ** (q / 5):.0f} photons")
    for a_, (m, lab) in zip(ax, panels):
        a_.set_xscale("log", base=2); a_.set_xticks([8, 16, 32, 64, 128, 256]); a_.set_xticklabels([8, 16, 32, 64, 128, 256])
        a_.set(xlabel="frames k", ylabel=lab); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_compression.{ext}", dpi=200)
    print(f"[aggregate] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
