#!/usr/bin/env python3
"""
Annealed plug-and-play HQS with non-blind learned denoisers (DPIR-style).
"""
from __future__ import annotations
import argparse, csv, itertools, json, os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon", _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import numpy as np
import torch

COHORT = [f"neural_cell_p{i}" for i in (64, 128, 192, 320, 448, 576, 704, 832)]
VAL = ["neural_cell_p303", "neural_cell_p10", "neural_cell_p693"]
DENS = {  # name -> list of (checkpoint, modes)
    "nc2d_nb": [("nc2d_nb", ("xy", "xz", "yz"))],
    "nc3d_nb": [("nc3d_nb", ("3d",))],
    "nc23_nb": [("nc2d_nb", ("xy", "xz", "yz")), ("nc3d_nb", ("3d",))],
}
GRID = {"c": [0.001, 0.01, 0.1], "s0": [0.1, 0.2], "s1": [0.01, 0.03]}


def cg(Aop, b, x0, iters):
    x = x0.clone(); r = b - Aop(x); p = r.clone(); rs = (r * r).sum()
    for _ in range(iters):
        Ap = Aop(p); a = rs / ((p * Ap).sum() + 1e-30); x += a * p; r -= a * Ap
        rn = (r * r).sum()
        if rn.sqrt() < 1e-6 * b.norm():
            break
        p = r + (rn / rs) * p; rs = rn
    return x


def hqs(A, y, shape, branches, c, s0, s1, K=20, cg_iters=20, gt=None, trace=None):
    from pipeline.learned_denoiser import apply_volume
    AtA = lambda v: A.t() @ (A @ v)
    v = torch.randn(A.shape[1], device=A.device)
    for _ in range(20):                                     # L = ||A||^2
        v = AtA(v); v /= v.norm()
    L = float((v * AtA(v)).sum())
    Aty = A.t() @ y
    z = Aty / (Aty.max() + 1e-12); x = z.clone()
    sig = np.geomspace(s0, s1, K)
    for i, s in enumerate(sig):
        mu = c * L * (s1 / s) ** 2
        x = cg(lambda u: AtA(u) + mu * u, Aty + mu * z, x, cg_iters)
        xv = x.reshape(shape); outs = []
        for model, modes in branches:
            for m in modes:
                outs.append(apply_volume(model, xv, m, float(s)))
        z = torch.stack(outs).mean(0).clamp_min(0).reshape(-1)
        if trace is not None and gt is not None:
            zc = z.reshape(shape).clamp(0, 1).cpu().numpy()
            trace.append({"ref_err": float(np.linalg.norm(zc - gt) / np.linalg.norm(gt)), "sigma": float(s)})
    return z.reshape(shape).clamp(0, 1).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["screen", "test"])
    ap.add_argument("--den", default="nc2d_nb"); ap.add_argument("--ks", default="16,64")
    ap.add_argument("--op", default="random1L_gap66"); ap.add_argument("--dir", default="results/recon/compression_1L_q15")
    ap.add_argument("--q", type=float, default=15); ap.add_argument("--K", type=int, default=20)
    a = ap.parse_args(); ks = [int(k) for k in a.ks.split(",")]; dev = "cuda"
    out = _ROOT / "results/paper/pnp_hqs" / a.den; out.mkdir(parents=True, exist_ok=True)
    from compression_subsets import Runner, rows_of
    from pipeline.learned_denoiser import load_model
    from pipeline.metrics import score
    from pipeline.metrics_paper import ssim3d_foreground
    r = Runner(argparse.Namespace(out=a.dir, snr=a.q, t_ref=256))
    T, Nd1, Nd2 = r.lab.meta[a.op]["meas_shape"]; rp = Nd1 * Nd2
    subs = json.loads((_ROOT / a.dir / "subsets.json").read_text())[a.op]
    Afull = torch.load(_ROOT / r.lab.meta[a.op]["A"], map_location="cpu", mmap=True)
    branches = [(load_model(_ROOT / "results/denoisers" / f"{ck}.pt", dev)[0], modes) for ck, modes in DENS[a.den]]
    cache = {}

    def problem(spec, k, seed):
        if k not in cache:
            rows = np.arange(T * rp) if k >= T else rows_of(subs[str(k)]["random"][0], rp)
            cache.clear(); cache[k] = (rows, Afull[torch.from_numpy(np.asarray(rows))].float().to(dev))
        rows, A = cache[k]
        y, _ = r.lab.measure(a.op, spec, r.snr_for(k, T, "total"), seed)
        return A, torch.from_numpy(np.asarray(y)[rows]).float().to(dev)

    def run(spec, k, seed, kn, trace=None):
        gt = np.asarray(r.lab.gt[spec]); A, y = problem(spec, k, seed); t = time.time()
        est = hqs(A, y, gt.shape, branches, kn["c"], kn["s0"], kn["s1"], K=a.K, gt=gt, trace=trace)
        m = score(gt, est); m["ssim3d_fg"] = ssim3d_foreground(gt, est); m["secs"] = round(time.time() - t, 1)
        return m, est

    log = out / "screen.jsonl"; bp = out / "best.json"; best = json.loads(bp.read_text()) if bp.exists() else {}
    if a.stage == "screen":
        seen = [json.loads(l) for l in open(log)] if log.exists() else []
        for k in ks:
            tried = {tuple(sorted(j["knobs"].items())): j["psnr"] for j in seen if j["k"] == k}
            def go(kn):
                key = tuple(sorted(kn.items()))
                if key in tried or kn["s1"] >= kn["s0"]:
                    return
                ms = [run(sp, k, 100, kn)[0] for sp in VAL]
                m = {q: float(np.mean([x[q] for x in ms])) for q in ("psnr", "local_depth_err", "ssim3d_fg", "secs")}
                tried[key] = m["psnr"]
                with open(log, "a") as f:
                    f.write(json.dumps({"k": k, "knobs": kn, **m}) + "\n")
                print(f"[hqs {a.den}] k={k:3d} {kn}: PSNR {m['psnr']:.2f} depth {m['local_depth_err']:.2f} fg {m['ssim3d_fg']:.3f} "
                      f"({m['secs']}s/vol)", flush=True)
            names = list(GRID)
            for combo in itertools.product(*[GRID[n] for n in names]):
                go(dict(zip(names, combo)))
            for _ in range(5):
                b = dict(max(tried, key=tried.get)); moved = False
                for n in names:
                    vals = sorted({dict(kk)[n] for kk in tried})
                    f = 10 if n == "c" else 2
                    if b[n] == vals[-1]:
                        go({**b, n: b[n] * f}); moved = True
                    elif b[n] == vals[0]:
                        go({**b, n: b[n] / f}); moved = True
                if not moved:
                    break
            bk = max(tried, key=tried.get); best[str(k)] = {"knobs": dict(bk), "val_psnr": tried[bk]}
            bp.write_text(json.dumps(best, indent=1))
            print(f"--- k={k}: best {dict(bk)} val PSNR {tried[bk]:.2f}", flush=True)
        return 0
    rows, traces = [], {}
    jobs = [("neural_cell", s) for s in (0, 1, 2)] + [(c, 0) for c in COHORT]
    for k in ks:
        kn = best[str(k)]["knobs"]
        for spec, seed in jobs:
            tr = []; m, est = run(spec, k, seed, kn, trace=tr); traces[f"{spec}_k{k}_s{seed}"] = tr
            rows.append({"den": a.den, "k": k, "specimen": spec, "seed": seed, "knobs": json.dumps(kn), "psnr": m["psnr"],
                         "depth_um": m["local_depth_err"] * 0.64, "fg_ssim": m["ssim3d_fg"], "secs": m["secs"]})
            np.savez_compressed(out / f"vol_{spec}_k{k}_s{seed}.npz", est=est.astype(np.float32))
            print(f"[hqs test {a.den}] k={k:3d} {spec:17s} s{seed}: PSNR {m['psnr']:.2f} depth {m['local_depth_err'] * 0.64:.3f} um "
                  f"fg {m['ssim3d_fg']:.3f}", flush=True)
            with open(out / "test_rows.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "traces.json").write_text(json.dumps(traces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
