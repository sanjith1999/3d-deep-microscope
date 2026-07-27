#!/usr/bin/env python3
"""
Mismatch ablation report: paired differences between arms and per-iteration traces.
"""
from __future__ import annotations
import argparse, csv, json, statistics as st
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
MET = ["psnr", "local_depth_err_planes", "ssim3d_fg", "ssim3d", "resid_under_own_operator", "resid_common_fine_physics"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/paper/mismatch/final_1plate"); a = ap.parse_args()
    d = _ROOT / a.dir
    rows = [r for f in ("rows.csv", "rows_coarse.csv") if (d / f).exists() for r in csv.DictReader(open(d / f))]
    key = lambda r: (r["specimen"], int(r["k"]), int(r["seed"]))
    by = {}
    for r in rows:
        by.setdefault(key(r), {})[r["operator"]] = r
    paired = []
    for (s, k, seed), arms in sorted(by.items()):
        if "patch" not in arms:
            continue
        for other in ("point", "coarse"):
            if other in arms:
                paired.append({"object": s, "k": k, "noise_seed": seed, "comparison": f"patch - {other}",
                               **{f"d_{m}": float(arms["patch"][m]) - float(arms[other][m]) for m in MET}})
    with open(d / "paired_differences.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    summ = []
    for s in sorted({r["specimen"] for r in rows}):
        for op in ("patch", "point", "coarse"):
            for k in sorted({int(r["k"]) for r in rows}, reverse=True):
                g = [r for r in rows if r["specimen"] == s and r["operator"] == op and int(r["k"]) == k]
                if not g:
                    continue
                e = {"object": s, "operator": op, "k": k, "n_seeds": len(g)}
                for m in MET:
                    v = [float(r[m]) for r in g]; e[f"{m}_mean"] = st.mean(v); e[f"{m}_sd"] = st.stdev(v) if len(v) > 1 else 0.0
                summ.append(e)
    for e in summ:                                  # full-scan-to-4x difference within each arm
        f256 = [x for x in summ if x["object"] == e["object"] and x["operator"] == e["operator"] and x["k"] == 256]
        for m in ("psnr", "local_depth_err_planes", "ssim3d_fg"):
            e[f"d_{m}_vs_full"] = e[f"{m}_mean"] - f256[0][f"{m}_mean"] if f256 else ""
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    # ---- traces ----
    H = {}
    for f in ("histories.json", "histories_coarse.json"):
        if (d / f).exists():
            H.update(json.loads((d / f).read_text()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    flds = [("fit_own", "data fit, own operator"), ("fit_common_fine_physics", "data fit, fine physics"),
            ("ref_err_rel_l2_clipped", "rel. error to reference"), ("local_depth_err_planes_clipped", "depth error (planes)")]
    col = {"patch": "C0", "point": "C1", "coarse": "C3"}
    fig, ax = plt.subplots(2, 4, figsize=(11, 4.6), sharex=True)
    for i, s in enumerate(("neural_cell", "beads")):
        for j, (fld, lab) in enumerate(flds):
            for op in ("patch", "point", "coarse"):
                for k, ls in ((256, "-"), (64, "--")):
                    h = H.get(f"{s}_k{k}_s0_{op}", {}).get(fld)
                    if h:
                        ax[i, j].plot(range(1, len(h) + 1), h, ls, color=col[op], lw=1, label=f"{op}, {k} frames")
            ax[i, j].set_title(f"{'neuronal' if s == 'neural_cell' else 'beads'}: {lab}", fontsize=7)
            ax[i, j].tick_params(labelsize=6)
            if fld.startswith("fit"):
                ax[i, j].set_yscale("log")
        ax[i, 0].set_ylabel("value", fontsize=7)
    for j in range(4):
        ax[1, j].set_xlabel("ADMM iteration", fontsize=7)
    ax[0, 0].legend(fontsize=5, frameon=False)
    fig.suptitle("Per-iteration traces, noise seed 0, identical measurements and solver budget (20 ADMM, CG <= 20); "
                 "no convergence claim", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(d / f"fig_traces.{ext}", dpi=300, bbox_inches="tight")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    names = {"patch": "patch-integrated (ours)", "point": "point-sampled fine simulation", "coarse": "simulated at 0.64 µm"}
    for i, s in enumerate(("neural_cell", "beads")):
        for op in ("patch", "point", "coarse"):
            for k, ls in ((256, "-"), (64, "--")):
                runs = [H.get(f"{s}_k{k}_s{sd}_{op}", {}).get("ref_err_rel_l2_clipped") for sd in (0, 1, 2)]
                runs = [r for r in runs if r]
                if not runs:
                    continue
                m = [sum(v) / len(v) for v in zip(*runs)]
                ax[i].plot(range(1, len(m) + 1), m, ls, color=col[op], lw=1.2,
                           label=f"{names[op]}, {k} frames" if i == 0 else None)
                j = min(range(len(m)), key=lambda t: m[t])
                if op == "point" and j < len(m) - 1:
                    ax[i].plot(j + 1, m[j], "o", ms=3, color=col[op])
        ax[i].set_title("neuronal subvolume 0" if s == "neural_cell" else "synthetic bead phantom", fontsize=8)
        ax[i].set_xlabel("ADMM iteration", fontsize=7); ax[i].tick_params(labelsize=6)
    ax[0].set_ylabel(r"$\|\hat x - x\|_2 / \|x\|_2$", fontsize=8)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Error to the reference during reconstruction (mean of 3 noise seeds; identical measurements and solver budget; "
                 "dot = minimum of the point-sampled arm)", fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(d / f"fig_error_vs_iteration.{ext}", dpi=300, bbox_inches="tight")
    print(f"[mismatch_report] {len(paired)} paired rows, {len(summ)} summary rows -> {d}")


if __name__ == "__main__":
    main()
