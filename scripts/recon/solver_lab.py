#!/usr/bin/env python3
"""
Solver lab: screen, tune and test PnP variants on the same operator, measurements and noise draws.

    python -u scripts/recon/solver_lab.py screen
    python -u scripts/recon/solver_lab.py test --configs base,t23n --val nc --seeds 0,1,2
"""
from __future__ import annotations
import argparse, csv, itertools, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon", _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import numpy as np

COHORT = [f"neural_cell_p{i}" for i in (64, 128, 192, 320, 448, 576, 704, 832)]
CONFIGS = {
    "base":  ({}, {"rhos": [0.5, 1, 2]}),
    "nn":    ({"nonneg": True}, {"rhos": [0.5, 1, 2]}),
    "17L":   ({"variant": "17L"}, {"rhos": [0.5, 1, 2]}),
    "17H":   ({"variant": "17H"}, {"rhos": [0.5, 1, 2]}),
    "n17":   ({"variant": "17N"}, {"rhos": [0.5, 1, 2], "lmda": [0.0125, 0.025, 0.05, 0.1]}),
    "n17nn": ({"variant": "17N", "nonneg": True}, {"rhos": [0.5, 1, 2], "lmda": [0.0125, 0.025, 0.05, 0.1]}),
    "bm4d":  ({"lmda": 0.0, "nonneg": True}, {"rhos": [0.5, 1, 2], "bm4d_lmda": [0.005, 0.02, 0.08]}),
    "hyb":   ({"nonneg": True}, {"rhos": [0.5, 1, 2], "bm4d_lmda": [0.005, 0.02, 0.08]}),
    # trained domain denoisers (scripts/denoise/train_denoiser.py; results/denoisers/<name>.pt)
    "t2n":   ({"td2": "nc2d"}, {"rhos": [0.5, 1, 2]}),                                  # neuronal 2D, three slice branches
    "t3n":   ({"td3": "nc3d", "lmda": 0.0}, {"rhos": [0.5, 1, 2]}),                     # neuronal 3D, single branch
    "t23n":  ({"td2": "nc2d", "td3": "nc3d"}, {"rhos": [0.5, 1, 2]}),                   # both
    "t2v":   ({"td2": "bv2d"}, {"rhos": [0.5, 1, 2]}),                                  # vasculature-trained 2D
    "t3v":   ({"td3": "bv3d", "lmda": 0.0}, {"rhos": [0.5, 1, 2]}),                     # vasculature-trained 3D
    "t2u3":  ({"td2": "nc2d", "td3": "nc3du"}, {"rhos": [0.5, 1, 2, 4]}),               # 2D slices + 3D U-Net
    "t23v":  ({"td2": "bv2d", "td3": "bv3d"}, {"rhos": [0.5, 1, 2, 4]}),                # VASCULATURE-trained 2D + 3D (cross-domain)
}
DEN = _ROOT / "results/denoisers"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["screen", "test"])
    ap.add_argument("--op", default="random1L_gap66"); ap.add_argument("--dir", default="results/recon/compression_1L_q15")
    ap.add_argument("--ks", default="16,64"); ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--q", type=float, default=15)
    ap.add_argument("--specs", default=None, help="test only: specimen list (seed 0) instead of subvolume 0 x3 + cohort")
    ap.add_argument("--save", action="store_true", help="test only: save reconstructed volumes (npz)")
    ap.add_argument("--draws", default="0", help="test only: frame-subset draws (e.g. 0,1,2,3,4); k = 256 always uses all")
    ap.add_argument("--out_name", default=None, help="test only: output csv name (so runs do not overwrite each other)")
    ap.add_argument("--seeds", default=None, help="test only: noise seeds for every specimen (e.g. 0,1,2); default: subvolume 0 x3, cohort x1")
    ap.add_argument("--val", default="vasc", help="'vasc' (paper: vasculature, seed 100) or 'nc' (held-out, overlap-cleared "
                    "neuronal crops p303, p10, p693, seed 100; never trained on, never tested)")
    a = ap.parse_args(); ks = [int(k) for k in a.ks.split(",")]; cfgs = a.configs.split(",")
    out = _ROOT / "results/paper/solver_lab" / a.op; out.mkdir(parents=True, exist_ok=True)
    import scico.numpy as snp
    from compression_subsets import Runner, rows_of, TUNE_ON
    from pipeline.reconstruct import run_pnp
    from pipeline.metrics import score
    from pipeline.metrics_paper import ssim3d_foreground
    r = Runner(argparse.Namespace(out=a.dir, snr=a.q, t_ref=256))
    T, Nd1, Nd2 = r.lab.meta[a.op]["meas_shape"]; rp = Nd1 * Nd2
    subs = json.loads((_ROOT / a.dir / "subsets.json").read_text())[a.op]
    cacheA = {}

    def problem(spec, k, seed, draw=0):
        if (k, draw) not in cacheA:
            rows = np.arange(T * rp) if k >= T else rows_of(subs[str(k)]["random"][draw], rp)
            cacheA.clear(); cacheA[(k, draw)] = (rows, r.rows_A(a.op, rows))
        rows, A = cacheA[(k, draw)]
        y, _ = r.lab.measure(a.op, spec, r.snr_for(k, T, "total"), seed)
        return A, snp.array(y[rows])

    def prior(cfg, k, knobs):
        st = {**r.tuned("dncnn", min(k, T)), "rhos": r.tuned("dncnn", min(k, T))["rhos"] * knobs.get("rhos", 1)}
        p = r.prior("dncnn", st); fixed, _ = CONFIGS[cfg]
        if "variant" in fixed:
            p["variant"] = fixed["variant"]
        if "lmda" in fixed:
            p["lmda"] = fixed["lmda"]
        if "lmda" in knobs:
            p["lmda"] = knobs["lmda"]
        if "bm4d_lmda" in knobs:
            p["bm4d_lmda"] = knobs["bm4d_lmda"]; p["bm4d_rho"] = p["rho"]
        if fixed.get("nonneg"):
            p["nonneg_rho"] = p["rho"]
        if "td2" in fixed:
            p["td2"] = str(DEN / f"{fixed['td2']}.pt")
        if "td3" in fixed:
            p["td3"] = str(DEN / f"{fixed['td3']}.pt"); p["td3_lmda"] = 1.0; p["td3_rho"] = p["rho"]
        return p

    def solve(A, y, gt, p, keep=None):
        t = time.time(); est, _ = run_pnp(A, y, obj_shape=gt.shape, gt=None, **p); est = np.asarray(est)
        m = score(gt, est); m["ssim3d_fg"] = ssim3d_foreground(gt, est); m["secs"] = round(time.time() - t, 1)
        if keep is not None:
            keep.append(est)
        return m

    VAL = {"vasc": [TUNE_ON["dncnn"][0]], "nc": ["neural_cell_p303", "neural_cell_p10", "neural_cell_p693"]}[a.val]
    tag = "" if a.val == "vasc" else f"_{a.val}"                       # vasculature keeps the original file names
    log = out / f"screen{tag}.jsonl"; best_p = out / f"best{tag}.json"
    best = json.loads(best_p.read_text()) if best_p.exists() else {}
    if a.stage == "screen":
        seed = TUNE_ON["dncnn"][1]
        seen = [json.loads(l) for l in open(log)] if log.exists() else []
        for k in ks:
            probs = [(r.lab.gt[sp], *problem(sp, k, seed)) for sp in VAL]
            for cfg in cfgs:
                _, grid = CONFIGS[cfg]
                tried = {tuple(sorted(j["knobs"].items())): j["psnr"] for j in seen if j["cfg"] == cfg and j["k"] == k}
                def run(knobs):
                    key = tuple(sorted(knobs.items()))
                    if key in tried:
                        return
                    try:
                        ms = [solve(A, y, gt, prior(cfg, k, knobs)) for (gt, A, y) in probs]
                        m = {q: float(np.mean([mm[q] for mm in ms])) for q in ("psnr", "local_depth_err", "ssim3d_fg", "secs")}
                    except Exception as e:                          # a config that cannot run is reported, not fatal
                        print(f"[lab] {cfg} k={k} {knobs}: FAILED {e!r}", flush=True); tried[key] = float("-inf"); return
                    tried[key] = m["psnr"]
                    j = {"cfg": cfg, "k": k, "knobs": knobs, "psnr": m["psnr"], "depth_planes": m["local_depth_err"],
                         "fg_ssim": m["ssim3d_fg"], "secs": m["secs"]}
                    with open(log, "a") as f:
                        f.write(json.dumps(j) + "\n")
                    print(f"[lab] {cfg:5s} k={k:3d} {knobs}: PSNR {m['psnr']:.2f} depth {m['local_depth_err']:.2f} "
                          f"fg {m['ssim3d_fg']:.3f} ({m['secs']}s)", flush=True)
                names = list(grid)
                for combo in itertools.product(*[grid[n] for n in names]):
                    run(dict(zip(names, combo)))
                for _ in range(4):                                   # extend each knob past an edge
                    b = dict(max(tried, key=tried.get)); moved = False
                    for n in names:
                        vals = sorted({dict(kk)[n] for kk in tried})
                        if b[n] == vals[-1]:
                            run({**b, n: b[n] * 2}); moved = True
                        elif b[n] == vals[0] and b[n] > 0:
                            run({**b, n: b[n] / 2}); moved = True
                    if not moved:
                        break
                bk = max(tried, key=tried.get)
                best[f"{cfg}_{k}"] = {"knobs": dict(bk), "val_psnr": tried[bk]}
                best_p.write_text(json.dumps(best, indent=1))
        # ranked summary
        for k in ks:
            print(f"--- k={k} (validation PSNR, best knobs) ---")
            rows = sorted(((c, best[f"{c}_{k}"]) for c in cfgs if f"{c}_{k}" in best), key=lambda t: -t[1]["val_psnr"])
            for c, b in rows:
                print(f"  {c:6s} {b['val_psnr']:.2f}  {b['knobs']}")
        return 0
    # ---- test ----
    rows = []
    if a.seeds:
        sl = [int(x) for x in a.seeds.split(",")]
        jobs = [(sp, sd) for sp in (a.specs.split(",") if a.specs else ["neural_cell"] + COHORT) for sd in sl]
    else:
        jobs = [(s, 0) for s in a.specs.split(",")] if a.specs else [("neural_cell", s) for s in (0, 1, 2)] + [(c, 0) for c in COHORT]
    vol_dir = out / "volumes"; vol_dir.mkdir(exist_ok=True)
    draws = [int(x) for x in a.draws.split(",")]
    for k in ks:
      for draw in (draws if k < T else [0]):
        for spec, seed in jobs:
            gt = r.lab.gt[spec]; A, y = problem(spec, k, seed, draw)
            for cfg in cfgs:
                knobs = best.get(f"{cfg}_{k}", {}).get("knobs")
                if knobs is None:
                    print(f"[lab test] no validation-best knobs for {cfg} k={k}; skipped"); continue
                kp = [] if (a.save or spec == "beads") else None
                m = solve(A, y, gt, prior(cfg, k, knobs), keep=kp)
                if kp and a.save:
                    np.savez_compressed(vol_dir / f"{cfg}_{spec}_k{k}_s{seed}.npz", est=kp[0].astype(np.float32), gt=np.asarray(gt))
                row = {"cfg": cfg, "k": k, "draw": draw, "specimen": spec, "seed": seed, "knobs": json.dumps(knobs), "psnr": m["psnr"],
                       "depth_um": m["local_depth_err"] * 0.64, "fg_ssim": m["ssim3d_fg"], "secs": m["secs"]}
                if spec == "beads" and kp:                          # hallucination / ghost stress test
                    from pipeline.metrics import bead_report
                    from pipeline.metrics_paper import bead_localisation
                    from pipeline.specimens import BEADS, mirror_plane
                    br = bead_report(kp[0], BEADS, mirror_plane); row["ghost"] = float(np.mean([b["ghost_ratio"] for b in br]))
                    row.update({f"bead_{kk}": v for kk, v in bead_localisation(kp[0], BEADS, gt=np.asarray(gt)).items()})
                rows.append(row)
                print(f"[lab test] {cfg:5s} k={k:3d} {spec:17s} s{seed}: PSNR {m['psnr']:.2f} depth {m['local_depth_err'] * 0.64:.3f} um "
                      f"fg {m['ssim3d_fg']:.3f}", flush=True)
                fn = a.out_name or ("test_rows.csv" if not a.specs else "test_rows_specs.csv")
                with open(out / fn, "w", newline="") as f:
                    keys = sorted({kk for rr in rows for kk in rr}, key=lambda kk: list(rows[0]).index(kk) if kk in rows[0] else 99)
                    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
