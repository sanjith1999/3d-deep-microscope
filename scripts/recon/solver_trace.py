#!/usr/bin/env python3
"""
One fully resolved reconstruction with its ADMM iteration history.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon", _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import scico.numpy as snp
from scico import linop, loss, functional
from scico.optimize.admm import ADMM, LinearSubproblemSolver

from compression_subsets import Runner, preset_of, rows_of
from pipeline.reconstruct import RECON, DnCNN_YZ, DnCNN_ZX
from pipeline.metrics import score


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolved settings + ADMM trace of one reconstruction.")
    ap.add_argument("--op", default="random_gap66"); ap.add_argument("--specimen", default="neural_cell")
    ap.add_argument("--k", type=int, default=256); ap.add_argument("--q", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compression_out", default="results/recon/compression_q15")
    ap.add_argument("--out", default="results/paper/package/solver_trace.json")
    a = ap.parse_args()
    ra = argparse.Namespace(out=a.compression_out, snr=a.q, t_ref=256)
    r = Runner(ra)
    coder, T, rp = r.op_info(a.op)
    st = r.tuned(preset_of(a.specimen), a.k)
    over = r.prior(preset_of(a.specimen), st)
    cfg = {**RECON, **over}
    pats = list(range(T)) if a.k >= T else json.loads((Path(r.out) / "subsets.json").read_text())[a.op][str(a.k)]["random"][0]
    snr = r.snr_for(a.k, T, "total")
    y, _ = r.lab.measure(a.op, a.specimen, snr, a.seed)
    if len(pats) < T:
        rows = rows_of(pats, rp); A, y = r.rows_A(a.op, rows), y[rows]
    else:
        A = r.lab.A(a.op)
    gt = r.lab.gt[a.specimen]; shape = gt.shape
    Aop = linop.LinearOperator(input_shape=shape, output_shape=(A.shape[0],), eval_fn=lambda x: A @ x.flatten())
    f = loss.SquaredL2Loss(y=snp.array(y), A=Aop)
    pick = lambda key: cfg["lmda"] if cfg.get(key) is None else cfg[key]
    terms = [(pick("lmda_xy"), functional.DnCNN(variant=cfg["variant"]), cfg["rho"], "zx (default DnCNN)"),
             (pick("lmda_yz"), DnCNN_YZ(variant=cfg["variant"]), cfg["rho"], "xy (DnCNN_YZ)"),
             (pick("lmda_zx"), DnCNN_ZX(variant=cfg["variant"]), cfg["rho"], "yz (DnCNN_ZX)"),
             (cfg["l1_lmda"], functional.L1Norm(), cfg["l1_rho"], "L1")]
    terms = [t for t in terms if t[0] > 0]
    x0 = Aop.T @ snp.array(y); x0max = float(x0.max()); x0 = x0 / (x0max + 1e-12)
    solver = ADMM(f=f, g_list=[w * g for (w, g, _, _) in terms], C_list=[linop.Identity(shape)] * len(terms),
                  rho_list=[rr for (_, _, rr, _) in terms], x0=x0, maxiter=cfg["max_iter"],
                  subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cfg["f_tol"], "maxiter": cfg["f_iter"]}),
                  itstat_options={"display": True, "period": 1})
    t = time.time(); raw = np.asarray(solver.solve()); secs = time.time() - t
    hist = solver.itstat_object.history(transpose=True)
    est = np.clip(raw, 0, 1); m = score(gt, est)
    out = {"op": a.op, "specimen": a.specimen, "k": a.k, "noise_q": snr, "noise_rule": "sigma = mean(y) * 10^(-q/10), y = clean fine-grid measurement of these rows",
           "seed": a.seed, "patterns": pats if len(pats) < T else "all",
           "resolved_settings": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in cfg.items()},
           "tuned_from": {"specimen_preset": preset_of(a.specimen), "tuned": st, "file": f"{a.compression_out}/tune.json"},
           "terms": [{"weight": w, "rho": rr, "acts_on": lab} for (w, _, rr, lab) in terms],
           "data_term": "SquaredL2Loss: (1/2) ||y - A x||_2^2 (scico default scale)",
           "splitting": "ADMM, one identity split per prior term (C_i = I); scico initialises auxiliaries from x0 and scaled duals to 0",
           "x0": f"A^T y normalised by its maximum ({x0max:.6g})",
           "clipping": "final estimate only, to [0, 1]",
           "raw_min": float(raw.min()), "raw_max": float(raw.max()), "frac_outside_01": float(np.mean((raw < 0) | (raw > 1))),
           "secs": round(secs, 1), "metrics": {k: float(v) for k, v in m.items() if isinstance(v, (int, float))},
           "history_fields": list(hist._fields) if hasattr(hist, "_fields") else None,
           "history": {fld: [float(x) for x in getattr(hist, fld)] for fld in (hist._fields if hasattr(hist, "_fields") else [])}}
    p = _ROOT / a.out; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[trace] PSNR {m['psnr']:.2f}  raw min/max {raw.min():.3g}/{raw.max():.3g}  fields {out['history_fields']} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
