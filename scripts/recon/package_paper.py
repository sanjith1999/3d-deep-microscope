#!/usr/bin/env python3
"""
Package the manuscript evidence: manifest, configuration, tuned settings, operator hashes and tables.
"""
from __future__ import annotations
import csv, dataclasses, hashlib, json, platform, re, shutil, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from configs.default import get_config
from pipeline.metrics_paper import ssim3d, ssim3d_foreground
from pipeline.specimens import BEADS, BEADS_VAL, COHORT, STORED

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

R = _ROOT / "results"; OPS = R / "recon/ops"; OUT = R / "paper/package"
LAYER_OPS = [f"random{L}16_spread{s}" for L in ("", "1L") for s in ("", "_s1", "_s2")]


def sha256(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def jsonl(path):
    out = []
    for f in sorted(Path(path).glob("runs*.jsonl")):
        for l in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    return out


def views(v):
    return {"mip_xy": v.max(0), "mip_xz": v.max(2), "slice_xz": v[:, :, v.shape[2] // 2]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gts = {k: v.numpy() for k, v in torch.load(OPS / "specimens.pt")["gt"].items()}
    cfg = get_config()

    # ---------- manifest ----------
    import jax, scico, skimage, scipy
    ops_meta = {}
    for d in sorted(OPS.iterdir()):
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            m["A_sha256"] = sha256(_ROOT / m["A"]); ops_meta[d.name] = m
            print(f"[package] hashed {d.name}", flush=True)
    tuned = {d: json.loads((R / "recon" / d / "tune.json").read_text())
             for d in ("compression", "compression_q15") if (R / "recon" / d / "tune.json").exists()}
    manifest = {
        "source": "FRIS repository",
        "versions": {"python": platform.python_version(), "jax": jax.__version__, "scico": scico.__version__,
                     "numpy": np.__version__, "torch": torch.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__},
        "config_default": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                           for k, v in dataclasses.asdict(cfg).items()},
        "geometry_note": "element (n_layers phase layers, neuron pitch 0.32 um, spacing 3.3 um) -> 6.6 um gap -> object plane of a "
                         "4F relay -> vectorial NA-0.8 relay -> specimen; incoherent emission NA 0.1; 12x12 detector (pooling 8 on "
                         "the 96x96 analysis grid); reconstruction grid 16x48x48 at 0.64 um; illumination normalised to mean "
                         "dose_mean over the operator's own patterns (matched_dose)",
        "noise_rule": "additive Gaussian, sigma = mean(y_clean) * 10^(-q/10) per measurement vector (pipeline: reconlab.measure); "
                      "'total' dose: q_k = q + 5 log10(256/k). Not a calibrated photon count; q relates to a shot-noise-equivalent "
                      "N = 10^(q/5) counts per detector pixel per frame only as an interpretation.",
        "specimens": {"stored": STORED, "preprocessing": "stored volume (32x128x128) area-resampled to 32x96x96, max-normalised; "
                      "GT = 2x2x2 average pooling to 16x48x48", "cohort_test": COHORT, "beads": BEADS, "beads_val_tuning_only": BEADS_VAL,
                      "tuning": {"dncnn": "blood_cell, noise seed 100", "l1": "beads_val, noise seed 100"}, "test_noise_seeds": [0, 1, 2]},
        "tuned_settings": tuned,
        "solver": "pipeline/reconstruct.run_pnp: ADMM, SquaredL2Loss, identity splits, DnCNN 17M x3 (physical planes: default=zx, "
                  "DnCNN_YZ=xy, DnCNN_ZX=yz) + L1; LinearSubproblemSolver CG (tol f_tol, maxiter f_iter); x0 = A^T y / max; clip "
                  "[0,1] at the end. 17M ignores the proximal scale (solver_checks.json). See solver_trace.json for one resolved run.",
        "operators": ops_meta,
        "commands": {
            "compression": "scripts/recon/compression_subsets.py {select,tune,validate,run} [--snr q --out dir]",
            "fusion": "scripts/recon/paper_recon.py fusion --ops random_gap66 --specimens neural_cell --seeds 3 --q 15 --snr 15 "
                      "--compression_out results/recon/compression_q15 --arms <arm list> --out results/paper/fusion_{a,b,c}",
            "information": "scripts/recon/point_information.py [--coders random2,random1,dmd | random1p,random2p --tag _paraxial]",
            "structure_metrics": "scripts/recon/analyse_volumes.py --dir <dir> [--redo]",
            "tables": "scripts/recon/aggregate_paper.py --dirs <dir:q,...>",
            "trace": "scripts/recon/solver_trace.py"}}
    for f in ("solver_checks.json",):
        if (R / "recon" / f).exists():
            manifest[f] = json.loads((R / "recon" / f).read_text())
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # ---------- layers ----------
    (OUT / "layers/volumes").mkdir(parents=True, exist_ok=True)
    rows = []
    for d, q in (("compression", 25), ("compression_q15", 15)):
        for r in jsonl(R / "recon" / d):
            if r["op"] in LAYER_OPS and r["k"] == 16:
                m = re.match(r"random(1L)?16_spread(?:_s(\d))?", r["op"])
                rows.append({"q": q, "op": r["op"], "layers": 1 if m.group(1) else 2, "element_seed": int(m.group(2) or 0),
                             "noise_seed": r["seed"], "specimen": r["specimen"], "k": 16, "snr_used": r["snr"],
                             "psnr": r["psnr"], "local_depth_err_planes": r["local_depth_err"],
                             "local_depth_err_um": r["local_depth_err"] * 0.64, "ghost": r.get("ghost", ""),
                             "l1s": r.get("l1s"), "rhos": r.get("rhos"), "max_iter": r.get("max_iter")})
    if rows:
        with open(OUT / "layers/rows.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for d in ("compression", "compression_q15"):
        for f in (R / "recon" / d / "volumes").glob("random*16_spread*_k16_*_neural_cell_s0_*.npz"):
            shutil.copy(f, OUT / "layers/volumes" / f"{d}__{f.name}")
    np.savez_compressed(OUT / "layers/volumes/gt_neural_cell.npz", gt=gts["neural_cell"])

    # ---------- fusion ----------
    (OUT / "fusion/volumes").mkdir(parents=True, exist_ok=True)
    frows = []
    for d in sorted((R / "paper").glob("fusion_*")):
        p = d / "fusion_metrics.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            vf = d / "volumes" / f"{r['tag']}_{r['op']}_{r['specimen']}_s{r['seed']}.npz"
            if vf.exists():
                z = np.load(vf); gt = gts[r["specimen"]]
                r["ssim3d"] = ssim3d(gt, z["est"]); r["ssim3d_fg"] = ssim3d_foreground(gt, z["est"])
                shutil.copy(vf, OUT / "fusion/volumes" / vf.name)
            frows.append(r)
    if frows:
        keys = sorted({k for r in frows for k in r})
        with open(OUT / "fusion/rows.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(frows)

    # ---------- views ----------
    (OUT / "views").mkdir(exist_ok=True)
    sel = {"gt": gts["neural_cell"]}
    subs = json.loads((R / "recon/compression_q15/subsets.json").read_text())["random_gap66"]
    for k in (16, 64, 256):
        pats = subs[str(k)]["all" if k == 256 else "random"][0]
        h = hashlib.sha1(json.dumps(sorted(int(p) for p in pats)).encode()).hexdigest()[:10]
        hits = sorted((R / "recon/compression_q15/volumes").glob(f"random_gap66_k{k}_{h}_neural_cell_s0_*.npz"))
        if hits:
            sel[f"k{k}"] = np.load(hits[0])["est"].astype(np.float32)
    arr = {f"{n}__{v}": a for n, vol in sel.items() for v, a in views(vol).items()}
    np.savez_compressed(OUT / "views/neural_cell_q15_views.npz", **arr)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(sel); vmax = float(max(v.max() for v in sel.values()))
    fig, ax = plt.subplots(2, len(names), figsize=(2.6 * len(names), 4.4), squeeze=False)
    for j, n in enumerate(names):
        ax[0, j].imshow(sel[n].max(0), cmap="magma", vmin=0, vmax=vmax); ax[0, j].set_title(n, fontsize=9)
        ax[1, j].imshow(sel[n].max(2), cmap="magma", vmin=0, vmax=vmax, aspect=1.0)
        for a_ in ax[:, j]:
            a_.set_xticks([]); a_.set_yticks([])
    ax[0, 0].set_ylabel("xy MIP"); ax[1, 0].set_ylabel("xz MIP (z down)")
    fig.suptitle("neural_cell, q = 15, noise seed 0, random_gap66 (2-layer); shared colour limits [0, %.2f]; 0.64 um voxels" % vmax, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "views/neural_cell_q15_views.png", dpi=200)

    if (R / "paper/tables").exists():
        shutil.copytree(R / "paper/tables", OUT / "tables", dirs_exist_ok=True)
    print(f"[package] -> {OUT}  (layers {len(rows)} rows, fusion {len(frows)} rows, views {list(sel)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
