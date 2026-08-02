#!/usr/bin/env python3
"""
Cohort table over the eight test crops: draws averaged per crop, then mean and SD across crops.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, statistics as st

COHORT = [f"neural_cell_p{i}" for i in (64, 128, 192, 320, 448, 576, 704, 832)]   # pipeline.specimens.COHORT
PLANE_UM = 0.64


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="256,64,32,16"); a = ap.parse_args(); ks = [int(k) for k in a.ks.split(",")]
    runs = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.dir, "runs*.jsonl"))) for l in open(f) if l.strip()]
    vm = list(csv.DictReader(open(os.path.join(a.dir, "volume_metrics.csv"))))
    per = {}   # (crop, k) -> dict of per-draw lists
    for c in COHORT:
        for k in ks:
            rr = [r for r in runs if r["specimen"] == c and r["k"] == k and r["seed"] == 0]
            vv = [v for v in vm if v["specimen"] == c and int(v["k"]) == k and int(v["seed"]) == 0]
            if rr:
                per[(c, k)] = {"depth": [r["local_depth_err"] * PLANE_UM for r in rr], "psnr": [r["psnr"] for r in rr],
                               "fg": [float(v["ssim3d_fg"]) for v in vv]}
    mean = lambda x: sum(x) / len(x)
    sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
    rows, crop_rows = [], []
    for k in ks:
        crops = [c for c in COHORT if (c, k) in per and (c, 256) in per]
        if not crops:
            continue
        m = {q: [mean(per[(c, k)][q]) for c in crops] for q in ("depth", "psnr", "fg")}
        d = {q: [mean(per[(c, k)][q]) - mean(per[(c, 256)][q]) for c in crops] for q in ("depth", "psnr", "fg")}
        dr = {q: [sd(per[(c, k)][q]) for c in crops] for q in ("depth", "psnr", "fg")}
        rows.append({"k": k, "compression": 256 // k, "n_crops": len(crops), "draws_per_crop": len(per[(crops[0], k)]["depth"]),
                     "depth_um_mean": mean(m["depth"]), "depth_um_sd_across_crops": sd(m["depth"]),
                     "fg_ssim_mean": mean(m["fg"]), "fg_ssim_sd_across_crops": sd(m["fg"]),
                     "psnr_mean": mean(m["psnr"]), "psnr_sd_across_crops": sd(m["psnr"]),
                     "d_depth_um_mean": mean(d["depth"]), "d_depth_um_sd_across_crops": sd(d["depth"]),
                     "d_fg_ssim_mean": mean(d["fg"]), "d_fg_ssim_sd_across_crops": sd(d["fg"]),
                     "d_psnr_mean": mean(d["psnr"]), "d_psnr_sd_across_crops": sd(d["psnr"]),
                     "draw_sd_depth_um_mean_over_crops": mean(dr["depth"]), "draw_sd_fg_ssim_mean_over_crops": mean(dr["fg"]),
                     "draw_sd_psnr_mean_over_crops": mean(dr["psnr"])})
        for c, a1, a2, a3, b1, b2, b3 in zip(crops, m["depth"], m["fg"], m["psnr"], d["depth"], d["fg"], d["psnr"]):
            crop_rows.append({"crop": c, "k": k, "depth_um_drawmean": a1, "fg_ssim_drawmean": a2, "psnr_drawmean": a3,
                              "d_depth_um": b1, "d_fg_ssim": b2, "d_psnr": b3})
    os.makedirs(a.out, exist_ok=True)
    for name, data in (("cohort_crop_level.csv", rows), ("cohort_crop_means.csv", crop_rows)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    with open(os.path.join(a.out, "cohort_crop_level.md"), "w", encoding="utf-8") as f:
        f.write("Cohort: the 8 prespecified subvolumes (p64, p128, p192, p320, p448, p576, p704, p832); noise seed 0; "
                "5 subset draws averaged within each crop first; mean ± sample SD (n−1) across the 8 crop means. "
                "Paired Δ = crop's k-frame draw mean − the same crop's full scan. Draw SD = mean over crops of the "
                "within-crop SD across draws (reported separately; not specimen variability). Excludes subvolume 0 and showcase crops.\n\n")
        f.write("| frames | C | depth error (µm) | fg SSIM | PSNR (dB) | Δdepth (µm) | Δfg SSIM | ΔPSNR (dB) | draw SD: depth / fg / PSNR |\n"
                "|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['k']} | {r['compression']}× | {r['depth_um_mean']:.3f} ± {r['depth_um_sd_across_crops']:.3f} | "
                    f"{r['fg_ssim_mean']:.3f} ± {r['fg_ssim_sd_across_crops']:.3f} | {r['psnr_mean']:.2f} ± {r['psnr_sd_across_crops']:.2f} | "
                    f"{r['d_depth_um_mean']:+.3f} ± {r['d_depth_um_sd_across_crops']:.3f} | {r['d_fg_ssim_mean']:+.3f} ± {r['d_fg_ssim_sd_across_crops']:.3f} | "
                    f"{r['d_psnr_mean']:+.2f} ± {r['d_psnr_sd_across_crops']:.2f} | {r['draw_sd_depth_um_mean_over_crops']:.3f} / "
                    f"{r['draw_sd_fg_ssim_mean_over_crops']:.3f} / {r['draw_sd_psnr_mean_over_crops']:.2f} |\n")
    print(open(os.path.join(a.out, "cohort_crop_level.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
