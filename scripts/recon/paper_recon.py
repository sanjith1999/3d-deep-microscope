#!/usr/bin/env python3
"""
Full-scan evidence runs with the tuned settings of the compression study.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon", _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import scico.numpy as snp

from compression_subsets import Runner, preset_of, rows_of, rule_random
from pipeline.reconstruct import run_pnp, RECON
from pipeline.metrics import score, bead_report, depth_blind_oracle
from pipeline.specimens import BEADS, mirror_plane

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = {"zx": "lmda_xy", "xy": "lmda_yz", "yz": "lmda_zx"}
ARMS = {"l1": ([], True), "xy+l1": (["xy"], True), "yz+l1": (["yz"], True), "zx+l1": (["zx"], True),
        "all+l1": (["xy", "yz", "zx"], True), "all": (["xy", "yz", "zx"], False)}


class Paper(Runner):
    def __init__(self, args):
        a = argparse.Namespace(**vars(args)); a.out = args.compression_out
        super().__init__(a)                                  # reads tune.json from the compression study
        self.pout = _ROOT / args.out; self.pout.mkdir(parents=True, exist_ok=True)

    def measure(self, op, rows, specimen, seed, q):
        y = self.lab._Yclean(op, specimen, "fine")
        y = y if rows is None else y[rows]
        if seed < 0:
            return y, np.zeros_like(y), 0.0
        sigma = float(y.mean()) * 10 ** (-q / 10)
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = sigma * torch.randn(y.shape, generator=g).numpy()
        return y + n, n, sigma

    def recon(self, op, pats, specimen, seed, q, prior, tag):
        coder, T, rp = self.op_info(op)
        rows = None if pats is None or len(pats) >= T else rows_of(pats, rp)
        A = self.lab.A(op) if rows is None else self.rows_A(op, rows)
        y, n, sigma = self.measure(op, rows, specimen, seed, q)
        gt = self.lab.gt[specimen]
        t = time.time()
        est_raw = np.asarray(run_pnp(A, snp.array(y), obj_shape=gt.shape, gt=None, **prior)[0])
        secs = time.time() - t
        est = np.clip(est_raw, 0, 1)
        m = score(gt, est)
        res = float(np.linalg.norm(np.asarray(A @ snp.array(est.reshape(-1))) - y) / (np.linalg.norm(y) + 1e-12))
        row = {"tag": tag, "op": op, "coder": coder, "specimen": specimen, "k": T if rows is None else len(pats),
               "seed": seed, "q": q, "sigma": sigma,
               "realised_snr_db_20log": (20 * math.log10(np.linalg.norm(y - n) / np.linalg.norm(n)) if seed >= 0 else float("inf")),
               "psnr_data_range": float(gt.max() - gt.min()), "raw_min": float(est_raw.min()), "raw_max": float(est_raw.max()),
               "frac_clipped": float(np.mean((est_raw < 0) | (est_raw > 1))), "rel_residual": res, "secs": round(secs, 1),
               "prior": json.dumps({k: v for k, v in prior.items()}),
               **{k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.integer, np.floating))}}
        row["local_depth_err_um"] = row["local_depth_err"] * 0.64
        if specimen == "beads":
            br = bead_report(est, BEADS, mirror_plane)
            row["ghost"] = float(np.mean([b["ghost_ratio"] for b in br]))
            row["beads"] = json.dumps(br)
        np.savez_compressed(self.pout / "volumes" / f"{tag}_{op}_{specimen}_s{seed}.npz",
                            gt=gt, est_raw=est_raw, est=est, err=est - gt)
        return row

    def write(self, name, rows):
        p = self.pout / name; new = not p.exists()
        keys = sorted({k for r in rows for k in r})
        if not new:
            with open(p, newline="", encoding="utf-8") as f:
                keys = next(csv.reader(f)) or keys
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)

    def done(self, name):
        p = self.pout / name
        if not p.exists():
            return set()
        with open(p, newline="", encoding="utf-8") as f:
            return {(r["tag"], r["op"], r["specimen"], int(r["seed"])) for r in csv.DictReader(f)}

    def full(self):
        (self.pout / "volumes").mkdir(exist_ok=True)
        have = self.done("full_metrics.csv")
        for spec in self.args.specimens:
            gt = self.lab.gt[spec]
            if ("oracle", "-", spec, 0) not in have:
                m = score(gt, np.clip(depth_blind_oracle(gt), 0, 1))
                self.write("full_metrics.csv", [{"tag": "oracle", "op": "-", "specimen": spec, "seed": 0,
                                                 **{k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating))}}])
            for op in self.args.ops:
                T = self.op_info(op)[1]
                prior = self.prior(preset_of(spec), self.tuned(preset_of(spec), T))
                for seed in [-1] + list(range(self.args.seeds)):
                    if ("full", op, spec, seed) in have:
                        continue
                    r = self.recon(op, None, spec, seed, self.args.q, prior, "full")
                    self.write("full_metrics.csv", [r])
                    print(f"[full] {op:14s} {spec:11s} seed {seed:2d}  PSNR {r['psnr']:.2f}  depthErr {r['local_depth_err']:.2f}"
                          + (f"  ghost {r['ghost']:.2f}" if "ghost" in r else "") + f"  ({r['secs']}s)", flush=True)

    def fusion(self):
        (self.pout / "volumes").mkdir(exist_ok=True)
        have = self.done("fusion_metrics.csv")
        for op in self.args.ops:
            coder, T, _ = self.op_info(op)
            k = self.args.k or T
            pats = None if k >= T else rule_random(k, T, 0)
            st = self.tuned("dncnn", k)                         # shared L1 weight, penalty, iterations
            base = self.prior("dncnn", st)
            for spec in self.args.specimens:
                for arm in self.args.arms:
                    planes, use_l1 = ARMS[arm]
                    prior = dict(base)
                    for plane, key in KEY.items():
                        prior[key] = RECON["lmda"] if plane in planes else 0.0
                    if not use_l1:
                        prior["l1_lmda"] = 0.0
                    tag = f"fusion-{arm}-k{k}"
                    for seed in range(self.args.seeds):
                        if (tag, op, spec, seed) in have:
                            continue
                        r = self.recon(op, pats, spec, seed, self.args.q, prior, tag)
                        r["arm"] = arm
                        self.write("fusion_metrics.csv", [r])
                        print(f"[fusion] {op} k={k} {arm:7s} {spec:11s} seed {seed}  PSNR {r['psnr']:.2f}  "
                              f"depthErr {r['local_depth_err']:.2f}" + (f"  ghost {r['ghost']:.2f}" if "ghost" in r else "")
                              + f"  ({r['secs']}s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-scan matrix and fusion ablation for the paper.")
    ap.add_argument("stage", choices=["full", "fusion"])
    ap.add_argument("--ops", default="random_gap66")
    ap.add_argument("--specimens", default="neural_cell,blood_cell,beads")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--q", type=float, default=25.0, help="noise level q: sigma = mean(y) * 10^(-q/10)")
    ap.add_argument("--k", type=int, default=0, help="fusion: frames (0 = full scan)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--compression_out", default="results/recon/compression")
    ap.add_argument("--out", default="results/paper/recon")
    # Runner needs these attributes; values unused here
    ap.add_argument("--snr", type=float, default=25.0); ap.add_argument("--t_ref", type=int, default=256)
    a = ap.parse_args()
    a.ops = a.ops.split(","); a.specimens = a.specimens.split(","); a.arms = a.arms.split(",")
    p = Paper(a)
    getattr(p, a.stage)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
