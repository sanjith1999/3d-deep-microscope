#!/usr/bin/env python3
"""
Detection NA x detector sampling x number of scan frames.

NA in {0.1, 0.2, 0.3} and pooling d in {8, 4, 2} (12x12 / 24x24 / 48x48 pixels) with 256 / 64 / 16 frames, so the
full level has M = 36,864 measurements; the eighth level keeps 1/8 of the frames. Noise follows equal detected
photons: q = 15 + 5 log10(36864 / M).

    python -u scripts/recon/detector_study.py build
    python -u scripts/recon/detector_study.py tune
    python -u scripts/recon/detector_study.py test
"""
from __future__ import annotations
import argparse, csv, json, os, random, subprocess, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon", _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import numpy as np

COHORT = [f"neural_cell_p{i}" for i in (64, 128, 192, 320, 448, 576, 704, 832)]
VAL = ["neural_cell_p303", "neural_cell_p10", "neural_cell_p693"]
M_REF = 36864
FRAMES = {8: 256, 4: 64, 2: 16, 1: 4}
CONFIGS = [(na, d) for na in (0.1, 0.2, 0.3) for d in (8, 4, 2)]   # d = 1 (96x96) dropped: ~11 GB footprint tensor,
                                                                    # and past Nyquist even at NA 0.3 (0.44 um needed)
DEN = _ROOT / "results/denoisers"


def opname(na, d):
    return f"det_na{int(round(na * 10)):02d}_d{d}"


def frames_of(d):
    T = FRAMES[d]
    return list(range(256)) if T == 256 else sorted(random.Random(0).sample(range(256), T))


def level_frames(d, level):
    T = FRAMES[d]
    if level == "full":
        return list(range(T))
    n = max(1, T // 8)
    return sorted(random.Random(1).sample(range(T), n))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["build", "tune", "test"])
    ap.add_argument("--configs", default=None, help="comma list of op names (default: all)")
    ap.add_argument("--levels", default="full,eighth"); ap.add_argument("--solvers", default="base,t23n")
    ap.add_argument("--out_name", default=None, help="test output csv name (default test_rows[_<designed>].csv)")
    ap.add_argument("--designed", default=None, help="build/tune/test a TRAINED plate: 'name:na:d:ckpt' (ckpt under results/designs), or 'name:na:d:random<seed>'")
    a = ap.parse_args()
    cfgs = [c for c in CONFIGS if a.configs is None or opname(*c) in a.configs.split(",")]
    designed = None
    if a.designed:                                                     # one extra operator with a trained plate
        dn, dna, dd, dck = a.designed.split(":"); designed = (dn, float(dna), int(dd), dck)
        cfgs = [(float(dna), int(dd))]
        global opname
        _orig = opname
        opname = lambda na, d: dn if (na, d) == (float(dna), int(dd)) else _orig(na, d)
    out = _ROOT / "results/paper/detector_study"; out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    if a.stage == "build":
        for na, d in cfgs:
            name = opname(na, d)
            if (_ROOT / "results/recon/ops" / name / "meta.json").exists():
                print(f"[det] {name} exists"); continue
            pf = out / f"patterns_d{d}.json"; pf.write_text(json.dumps(frames_of(d)))
            t = time.time()
            if designed and designed[3].startswith("random"):              # 'random<seed>': another untrained plate (seed check)
                coder = ["--coder", "random", "--seed", designed[3][6:], "--n_layers", "1"]
            elif designed:
                coder = ["--coder", "designed", "--designed", str(_ROOT / "results/designs" / designed[3])]
            else:
                coder = ["--coder", "random", "--seed", "0", "--n_layers", "1"]
            subprocess.check_call([py, "-u", str(_ROOT / "scripts/recon/build_fast.py"), *coder, "--name", name,
                                   "--d_size", str(d), "--na_em", str(na), "--pattern_idx", str(pf)], cwd=_ROOT)
            subprocess.check_call([py, "-u", str(_ROOT / "scripts/recon/add_specimen.py"), "--ops", name,
                                   "--specimen", ",".join(COHORT + VAL)], cwd=_ROOT)
            print(f"[det] {name} built + registered in {time.time() - t:.0f}s", flush=True)
        return 0
    import scico.numpy as snp
    from compression_subsets import Runner, rows_of
    from pipeline.reconstruct import run_pnp
    from pipeline.metrics import score
    from pipeline.metrics_paper import ssim3d_foreground
    r = Runner(argparse.Namespace(out="results/recon/compression_1L_q15", snr=15, t_ref=256))
    levels = a.levels.split(","); solvers = a.solvers.split(",")

    def problem(op, d, level, spec, seed):
        rp = (96 // d) ** 2; fr = level_frames(d, level); rows = rows_of(fr, rp); M = len(rows)
        q = 15 + 5 * np.log10(M_REF / M)
        y, _ = r.lab.measure(op, spec, q, seed)
        return r.rows_A(op, rows), snp.array(np.asarray(y)[rows]), M, q, len(fr)

    def prior(solver, rh):
        p = r.prior("dncnn", {"l1s": 0.0625, "rhos": rh, "max_iter": 20})
        if solver == "t23n":
            p["td2"] = str(DEN / "nc2d.pt"); p["td3"] = str(DEN / "nc3d.pt"); p["td3_lmda"] = 1.0; p["td3_rho"] = p["rho"]
        return p

    def solve(A, y, gt, p):
        est, _ = run_pnp(A, y, obj_shape=gt.shape, gt=None, **p); est = np.asarray(est)
        m = score(gt, est); m["fg"] = ssim3d_foreground(gt, est); return m

    bp = out / "best.json"; best = json.loads(bp.read_text()) if bp.exists() else {}
    log = out / "tune.jsonl"
    if a.stage == "tune":
        seen = [json.loads(l) for l in open(log)] if log.exists() else []
        for na, d in cfgs:
            op = opname(na, d)
            for level in levels:
                probs = None
                for solver in solvers:
                    key = f"{op}|{level}|{solver}"
                    tried = {j["rhos"]: j["psnr"] for j in seen if j["key"] == key}
                    def go(rh):
                        nonlocal probs
                        if rh in tried:
                            return
                        if probs is None:
                            probs = [(np.asarray(r.lab.gt[s]), *problem(op, d, level, s, 100)) for s in VAL]
                        ms = [solve(A, y, gt, prior(solver, rh)) for (gt, A, y, M, q, nf) in probs]
                        m = {f: float(np.mean([x[f] for x in ms])) for f in ("psnr", "local_depth_err", "fg")}
                        tried[rh] = m["psnr"]; M, q, nf = probs[0][3], probs[0][4], probs[0][5]
                        with open(log, "a") as f:
                            f.write(json.dumps({"key": key, "op": op, "na": na, "d": d, "level": level, "solver": solver, "rhos": rh,
                                                "M": M, "frames": nf, "q": q, **m}) + "\n")
                        print(f"[det tune] {key} frames {nf} M {M} rho x{rh}: PSNR {m['psnr']:.2f} depth {m['local_depth_err']:.2f} "
                              f"fg {m['fg']:.3f}", flush=True)
                    for rh in (0.25, 1, 4):
                        go(rh)
                    for _ in range(10):                              # refine / extend around the best (factor 2; 10 steps so NA 0.3 is interior)
                        b = max(tried, key=tried.get); R = sorted(tried); moved = False
                        for c in (b / 2, b * 2):
                            if c not in tried and (b in (R[0], R[-1]) or abs(np.log2(c / b)) == 1):
                                go(c); moved = True
                        if not moved:
                            break
                    b = max(tried, key=tried.get); R = sorted(tried)
                    best[key] = {"rhos": b, "val_psnr": tried[b], "edge": b in (R[0], R[-1])}
                    bp.write_text(json.dumps(best, indent=1))
        return 0
    rows_out = []
    jobs = [("neural_cell", s) for s in (0, 1, 2)] + [(c, 0) for c in COHORT]
    for na, d in cfgs:
        op = opname(na, d)
        for level in levels:
            for solver in solvers:
                key = f"{op}|{level}|{solver}"
                if key not in best:
                    print(f"[det test] {key}: not tuned; skipped"); continue
                for spec, seed in jobs:
                    gt = np.asarray(r.lab.gt[spec]); A, y, M, q, nf = problem(op, d, level, spec, seed)
                    m = solve(A, y, gt, prior(solver, best[key]["rhos"]))
                    rows_out.append({"op": op, "na": na, "d": d, "pixels": (96 // d) ** 2, "frames": nf, "M": M, "level": level,
                                     "solver": solver, "specimen": spec, "seed": seed, "q": q, "psnr": m["psnr"],
                                     "depth_um": m["local_depth_err"] * 0.64, "fg_ssim": m["fg"]})
                    print(f"[det test] {key} {spec:17s} s{seed}: PSNR {m['psnr']:.2f} depth {m['local_depth_err'] * 0.64:.3f} um "
                          f"fg {m['fg']:.3f}", flush=True)
                    with open(out / (a.out_name or (f"test_rows_{designed[0]}.csv" if designed else "test_rows.csv")), "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
