#!/usr/bin/env python3
"""
Compression in time: reconstruct from k of the 256 scan patterns under several subset rules.

Rules: first, random, grid, spread (farthest point in sine space), wiener (greedy predicted MSE).
Dose: 'frame' keeps per-frame SNR; 'total' shares the photons of the full scan.

    python -u scripts/recon/compression_subsets.py select
    python -u scripts/recon/compression_subsets.py tune
    python -u scripts/recon/compression_subsets.py run --ops random1L_gap66
    python -u scripts/recon/compression_subsets.py report
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import scico.numpy as snp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")             # Windows consoles/logs default to cp1252

from configs.default import get_config
from coder.diffractive import angle_scan_directions
from pipeline.reconstruct import run_pnp, RECON
from pipeline.metrics import score, bead_report
from pipeline.specimens import BEADS, mirror_plane
import reconlab

KS = (8, 16, 32, 64, 128, 256)
GRID = {8: (4, 2), 16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (16, 8), 256: (16, 16)}   # (n_alpha, n_theta)
RULES = ("first", "random", "grid", "spread", "wiener")
GEOMETRIC = ("grid", "spread", "wiener")             # need the tilt geometry / the angle-scan operator
PRESET = {"neural_cell": "dncnn", "blood_cell": "dncnn", "beads": "l1", "beads_val": "l1"}
TUNE_ON = {"dncnn": ("blood_cell", 100), "l1": ("beads_val", 100)}
TEST = ("neural_cell", "beads")


def preset_of(spec):
    return PRESET.get(spec, "l1" if spec.startswith("beads") else "dncnn")


# --- subset rules ---------------------------------------------------------------------------
def rule_first(k):
    return list(range(k))


def rule_random(k, T, draw):
    return sorted(np.random.default_rng(1000 + draw).choice(T, k, replace=False).tolist())


def rule_grid(k, na, nt):
    a, t = GRID[k]
    ia = ((np.arange(a) + 0.5) * na / a).astype(int)
    it = ((np.arange(t) + 0.5) * nt / t).astype(int)
    return sorted(int(i * nt + j) for i in ia for j in it)


def rule_spread(k, d):
    """Farthest-point sampling in sine space, starting from pattern 0."""
    sel, dist = [0], np.linalg.norm(d - d[0], axis=1)
    while len(sel) < k:
        i = int(dist.argmax()); sel.append(i)
        dist = np.minimum(dist, np.linalg.norm(d - d[i], axis=1))
    return sorted(sel)


class Wiener:

    def __init__(self, cols, rows_per, T, sigma, p0):
        n = cols.shape[1]
        B = cols.reshape(T, rows_per, n).astype(np.float64)
        self.P = np.einsum("tri,trj->tij", B, B) / sigma ** 2
        self.I = np.eye(n) / p0; self.n, self.T = n, T; self.sigma, self.p0 = sigma, p0

    def mse(self, S):
        return float(np.trace(np.linalg.inv(self.I + self.P[list(S)].sum(0)))) / self.n

    def greedy(self, kmax):
        """Greedy order; the first k entries are the k-subset (nested)."""
        order, M, left = [], self.I.copy(), list(range(self.T))
        for _ in range(kmax):
            tr = np.trace(np.linalg.inv(M[None] + self.P[left]), axis1=1, axis2=2)
            best = left.pop(int(tr.argmin())); order.append(best); M = M + self.P[best]
        return order


def lateral_laplacian(v):
    """-Laplacian over the two lateral axes of a (rz, rx, ry) volume (Neumann edges)."""
    out = 4 * v
    for ax in (1, 2):
        out = out - np.concatenate([v.take([0], ax), v.take(range(v.shape[ax] - 1), ax)], ax)
        out = out - np.concatenate([v.take(range(1, v.shape[ax]), ax), v.take([-1], ax)], ax)
    return out


def bayes_mse(A, sigma, p0, shape, beta=0.0, probes=16, iters=60, seed=0):
    n = int(np.prod(shape)); rng = np.random.default_rng(seed)
    def M(v):
        r = A.T @ (A @ v) / sigma ** 2 + v / p0
        if beta:
            r = r + beta / p0 * lateral_laplacian(v.reshape(shape)).reshape(-1)
        return r
    tot = 0.0
    for _ in range(probes):
        z = rng.choice([-1.0, 1.0], n).astype(np.float32)
        x = z * p0; r = z - M(x); q = r.copy(); rr = float(r @ r)       # CG on M x = z, start at p0*z
        for _ in range(iters):
            Mq = M(q); a = rr / float(q @ Mq); x += a * q; r -= a * Mq
            rn = float(r @ r)
            if rn < 1e-10 * n: break
            q = r + (rn / rr) * q; rr = rn
        tot += float(z @ x)
    return tot / probes / n


def rows_of(pats, rows_per):
    return np.concatenate([np.arange(p * rows_per, (p + 1) * rows_per) for p in sorted(pats)])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# --- solving --------------------------------------------------------------------------------
class Runner:
    def __init__(self, args):
        self.args = args
        self.lab = reconlab.Lab()
        self.out = _ROOT / args.out; self.out.mkdir(parents=True, exist_ok=True)
        self._mm = {}

    def rows_A(self, op, r):
        if op not in self._mm:
            self._mm[op] = torch.load(_ROOT / self.lab.meta[op]["A"], map_location="cpu", mmap=True)
        return snp.array(self._mm[op][torch.from_numpy(r)].float().numpy())

    def op_info(self, op):
        m = self.lab.meta[op]
        T, a, b = m["meas_shape"]
        return m["coder"], T, a * b

    DEFAULT = {"l1s": 1.0, "rhos": 1.0, "max_iter": RECON["max_iter"]}

    def combos(self):
        return [{"l1s": a, "rhos": b, "max_iter": it} for a in self.args.l1s for b in self.args.rhos
                for it in self.args.iters]

    def prior(self, preset, st):
        """Preset with the L1 weight scaled by st['l1s'] and both ADMM penalties (rho, l1_rho) by st['rhos']."""
        eff = {**RECON, **reconlab.PRESETS[preset]}
        over = dict(reconlab.PRESETS[preset])
        over["l1_lmda"] = eff["l1_lmda"] * st["l1s"]
        over["rho"] = eff["rho"] * st["rhos"]
        over["l1_rho"] = eff["l1_rho"] * st["rhos"]
        over["max_iter"] = int(st["max_iter"])
        return over

    def snr_for(self, k, T, dose):
        """'total': the photon budget of the full t_ref-frame scan shared by k frames (shot noise)."""
        return self.args.snr + (5 * math.log10(self.args.t_ref / k) if dose == "total" else 0.0)

    def solve(self, op, pats, specimen, seed, snr, prior):
        coder, T, rp = self.op_info(op)
        gt = self.lab.gt[specimen]
        y, _ = self.lab.measure(op, specimen, snr, seed)
        if len(pats) < T:
            r = rows_of(pats, rp)
            A, y = self.rows_A(op, r), y[r]
        else:
            A = self.lab.A(op)
        t = time.time()
        est = np.asarray(run_pnp(A, snp.array(y), obj_shape=gt.shape, gt=snp.array(gt), **prior)[0])
        m = score(gt, est); m["secs"] = round(time.time() - t, 1)
        if specimen == "beads":
            m["ghost"] = float(np.mean([b["ghost_ratio"] for b in bead_report(est, BEADS, mirror_plane)]))
        self.save_volume(est, op, len(pats), specimen, seed, snr, pats)
        return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.integer, np.floating))}

    def save_volume(self, est, op, k, specimen, seed, snr, pats):
        import hashlib
        h = hashlib.sha1(json.dumps(sorted(int(p) for p in pats)).encode()).hexdigest()[:10]
        d = self.out / "volumes"; d.mkdir(exist_ok=True)
        np.savez_compressed(d / f"{op}_k{k}_{h}_{specimen}_s{seed}_q{snr:.2f}.npz", est=est.astype(np.float16),
                            patterns=np.array(sorted(pats), dtype=np.int16))

    # --- jsonl helpers ---
    def lines(self, name):
        """Records of <name>.jsonl plus every worker's <name>_w*.jsonl."""
        stem = name[:-len(".jsonl")]
        files = [self.out / name] + sorted(self.out.glob(f"{stem}_w*.jsonl"))
        out = []
        for f in files:
            if f.exists():
                for l in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        out.append(json.loads(l))
                    except ValueError:
                        pass                                   # a line torn by a concurrent append
        return out

    def done(self, name, keyf):
        return {keyf(r) for r in self.lines(name)}

    def append(self, name, rec):
        with open(self.out / name, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def tuned(self, preset, k):
        p = self.out / "tune.json"
        if p.exists():
            t = json.loads(p.read_text()).get(preset, {}).get(str(k))
            if t:
                return {kk: t[kk] for kk in self.DEFAULT}
        return dict(self.DEFAULT)

    # --- select ---
    def wiener_for(self, op):
        coder, T, rp = self.op_info(op)
        spec, _ = TUNE_ON["dncnn"]
        g = self.lab.gt[spec].reshape(-1)
        ym = float(self.lab._Yclean(op, spec, "fine").mean())
        sigma = ym * 10 ** (-self.args.snr / 10)
        return Wiener(self.lab.columns(op), rp, T, sigma, float(g.var())), float(g.max() - g.min())

    def select(self):
        out = {}
        for op in self.args.ops:
            coder, T, rp = self.op_info(op)
            cfg = get_config()
            # tilt geometry only for a full angle-scan operator (a 16-pattern subset operator has none)
            geo = coder != "dmd" and cfg.n_alphas * cfg.n_thetas == T and any(k < T for k in self.args.ks)
            if geo:
                d = angle_scan_directions(cfg.n_alphas, cfg.n_thetas).numpy()
                t = time.time(); W, _ = self.wiener_for(op)
                order = W.greedy(max(k for k in self.args.ks if k < T))
                print(f"[select] {op}: wiener greedy in {time.time() - t:.0f}s", flush=True)
            sub = {}
            for k in self.args.ks:
                if k >= T:
                    sub[str(k)] = {"all": [list(range(T))]}; continue
                s = {"first": [rule_first(k)],
                     "random": [rule_random(k, T, i) for i in range(self.args.draws)]}
                if geo:
                    s["grid"] = [rule_grid(k, cfg.n_alphas, cfg.n_thetas)]
                    s["spread"] = [rule_spread(k, d)]
                    s["wiener"] = [sorted(order[:k])]
                    print(f"[select] {op} k={k:3d} predicted Wiener MSE: " + "  ".join(
                        f"{r} {np.mean([W.mse(x) for x in v]):.4g}" for r, v in s.items()), flush=True)
                sub[str(k)] = s
            out[op] = sub
        p = self.out / "subsets.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(out); p.write_text(json.dumps(old))
        print(f"[select] -> {p}")
        cfg = get_config()
        self.plot_subsets(out, angle_scan_directions(cfg.n_alphas, cfg.n_thetas).numpy())

    def plot_subsets(self, subs, d):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for op, sub in subs.items():
            ks = [k for k in sub if "all" not in sub[k]]
            rules = sorted({r for k in ks for r in sub[k]})
            if not ks or self.op_info(op)[0] == "dmd":          # DMD patterns have no tilt geometry
                continue
            fig, ax = plt.subplots(len(rules), len(ks), figsize=(2.2 * len(ks), 2.2 * len(rules)), squeeze=False)
            for j, k in enumerate(ks):
                for i, r in enumerate(rules):
                    a = ax[i, j]; a.scatter(d[:, 0], d[:, 1], s=3, c="0.85")
                    if r in sub[k]:
                        sel = sub[k][r][0]; a.scatter(d[sel, 0], d[sel, 1], s=9, c="C3")
                    a.set_aspect("equal"); a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_xticks([]); a.set_yticks([])
                    if i == 0: a.set_title(f"k={k}", fontsize=9)
                    if j == 0: a.set_ylabel(r + ("  (draw 0)" if r == "random" else ""), fontsize=9)
            fig.suptitle(f"{op}: scan directions s = cos(theta)(cos a, sin a) per subset rule", fontsize=10)
            fig.tight_layout(); fig.savefig(self.out / f"subsets_{op}.png", dpi=130); plt.close(fig)
            print(f"[select] -> {self.out / f'subsets_{op}.png'}")

    def subsets(self, op):
        p = self.out / "subsets.json"
        s = json.loads(p.read_text()) if p.exists() else {}
        if op not in s:
            raise SystemExit(f"no subsets for {op}: run 'select --ops {op}' first")
        return s[op]

    # --- tune ---
    def tune(self):
        op = self.args.ops[0]; coder, T, _ = self.op_info(op); subs = self.subsets(op)
        key = lambda r: (r.get("specimen"), r["preset"], r["k"], r["l1s"], r["rhos"], r["max_iter"])
        have = self.done("tune.jsonl", key)
        for preset, (spec, seed) in TUNE_ON.items():
            if self.args.tune_presets and preset not in self.args.tune_presets:
                continue
            for k in self.args.ks:
                pats = subs[str(k)]["all" if k >= T else "random"][0]
                for st in self.combos():
                    if (spec, preset, k, st["l1s"], st["rhos"], st["max_iter"]) in have:
                        continue                         # keyed by specimen: a tuning specimen change re-tunes
                    m = self.solve(op, pats, spec, seed, self.snr_for(k, T, self.args.tune_dose),
                                   self.prior(preset, st))
                    rec = {"op": op, "preset": preset, "specimen": spec, "seed": seed, "k": k, **st, **m}
                    self.append("tune.jsonl", rec)
                    print(f"[tune] {preset} k={k:3d} l1x{st['l1s']:<4g} rhox{st['rhos']:<4g} it={st['max_iter']:<3d} "
                          f"PSNR {m['psnr']:.2f} depthErr {m['local_depth_err']:.2f} ({m['secs']}s)", flush=True)
        best = {}
        for r in self.lines("tune.jsonl"):
            if r.get("specimen") != TUNE_ON[r["preset"]][0]:
                continue                                   # only the current tuning specimen counts
            b = best.setdefault(r["preset"], {}).get(str(r["k"]))
            if b is None or r["psnr"] > b["psnr"]:
                best[r["preset"]][str(r["k"])] = {**{kk: r[kk] for kk in self.DEFAULT}, "psnr": r["psnr"]}
        (self.out / "tune.json").write_text(json.dumps(best, indent=1))
        print(json.dumps(best, indent=1))

    # --- validate ---
    def validate(self):
        op = self.args.ops[0]; coder, T, _ = self.op_info(op); k = self.args.val_k
        W, peak = self.wiener_for(op)
        key = lambda r: (r["specimen"], r["i"])
        have = self.done("validate.jsonl", key)
        cache = {r["i"]: {kk: v for kk, v in r.items() if kk.startswith("pred")} for r in self.lines("validate.jsonl")}
        for spec in TEST:
            preset = preset_of(spec); st = self.tuned(preset, k)
            for i in range(self.args.n_val):
                if (spec, i) in have:
                    continue
                pats = sorted(np.random.default_rng(5000 + i).choice(T, k, replace=False).tolist())
                if i not in cache:                      # predictors do not depend on the specimen
                    c = {"pred": 10 * math.log10(peak ** 2 / W.mse(pats))}
                    if self.args.full_pred:
                        As = np.asarray(self.rows_A(op, rows_of(pats, self.op_info(op)[2])), dtype=np.float32)
                        for beta in self.args.betas:
                            t = time.time()
                            c[f"pred_full_b{beta:g}"] = 10 * math.log10(peak ** 2 / bayes_mse(
                                As, W.sigma, W.p0, self.lab.gt[spec].shape, beta=beta,
                                probes=self.args.pred_probes, iters=self.args.pred_iters))
                            print(f"[validate]   subset #{i} full-operator predictor beta={beta:g}: "
                                  f"{c[f'pred_full_b{beta:g}']:.2f} ({time.time() - t:.0f}s)", flush=True)
                    cache[i] = c
                pred = cache[i]["pred"]
                m = self.solve(op, pats, spec, 0, self.args.snr, self.prior(preset, st))
                self.append("validate.jsonl", {"op": op, "specimen": spec, "i": i, "k": k, **cache[i],
                                               "patterns": pats, **m})
                print(f"[validate] {spec} #{i:2d} pred {pred:.2f}  PSNR {m['psnr']:.2f}  "
                      f"depthErr {m['local_depth_err']:.2f} ({m['secs']}s)", flush=True)
        recs = self.lines("validate.jsonl")
        for spec in TEST:
            r = [x for x in recs if x["specimen"] == spec]
            if len(r) >= 3:
                print(f"[validate] {spec}: n={len(r)}  PSNR spread over random subsets "
                      f"{np.min([x['psnr'] for x in r]):.2f}..{np.max([x['psnr'] for x in r]):.2f}")
                for name in ["pred"] + sorted({kk for x in r for kk in x if kk.startswith("pred_full")}):
                    rr = [x for x in r if name in x]; p = [x[name] for x in rr]
                    print(f"[validate]   {name:16s} Spearman vs PSNR {spearman(p, [x['psnr'] for x in rr]):+.2f}"
                          f"   vs -depthErr {spearman(p, [-x['local_depth_err'] for x in rr]):+.2f}")
        print("[validate] pass bar: Spearman >= 0.5 on both specimens, else drop the 'wiener' rule from the claims")

    # --- run ---
    def run(self):
        key = lambda r: (r["op"], r["k"], r["rule"], r["draw"], r["specimen"], r["seed"], r["dose"])
        have = self.done("runs.jsonl", key)
        W, NW = self.args.worker, self.args.n_workers
        outname = "runs.jsonl" if NW == 1 else f"runs_w{W}.jsonl"
        task = -1
        for seed in range(self.args.seeds):                    # seed-major: seed 0 gives a first full picture
            for op in self.args.ops:
                coder, T, _ = self.op_info(op); subs = self.subsets(op)
                for k in self.args.ks:
                    for rule, draws in subs[str(k)].items():
                        if rule not in self.args.rules and rule != "all":
                            continue
                        for draw, pats in enumerate(draws):
                            for spec in self.args.specimens:
                                preset = preset_of(spec); st = self.tuned(preset, k)
                                for dose in self.args.doses:
                                    if k >= self.args.t_ref and dose == "total" and "frame" in self.args.doses:
                                        continue                # identical at k = t_ref
                                    task += 1
                                    if task % NW != W or (op, k, rule, draw, spec, seed, dose) in have:
                                        continue
                                    snr = self.snr_for(k, T, dose)
                                    m = self.solve(op, pats, spec, seed, snr, self.prior(preset, st))
                                    self.append(outname, {"op": op, "coder": coder, "k": k, "rule": rule,
                                                               "draw": draw, "specimen": spec, "seed": seed,
                                                               "dose": dose, "snr": snr, "M": k * self.op_info(op)[2],
                                                               **st, **m})
                                    print(f"[run] s{seed} {op} k={k:3d} {rule:6s}#{draw} {spec:11s} {dose:5s} "
                                          f"PSNR {m['psnr']:.2f} depthErr {m['local_depth_err']:.2f}"
                                          + (f" ghost {m['ghost']:.2f}" if "ghost" in m else "")
                                          + f" ({m['secs']}s)", flush=True)

    # --- detector binning: trade pixels for frames at a fixed measurement budget ---
    def binned(self, op, pats, b):
        coder, T, rp = self.op_info(op); nd = int(round(rp ** 0.5)); p = nd // b
        if op not in self._mm:
            self._mm[op] = torch.load(_ROOT / self.lab.meta[op]["A"], map_location="cpu", mmap=True)
        out = []
        for t in sorted(pats):
            blk = self._mm[op][t * rp:(t + 1) * rp].float().numpy()          # (144, n)
            out.append(blk.reshape(p, b, p, b, -1).mean((1, 3)).reshape(p * p, -1))
        return np.concatenate(out, 0)

    def bin_y(self, y, pats, rp, b):
        nd = int(round(rp ** 0.5)); p = nd // b
        Y = np.stack([y[t * rp:(t + 1) * rp] for t in sorted(pats)])          # (k, 144)
        return Y.reshape(-1, p, b, p, b).mean((2, 4)).reshape(-1)

    def solve_binned(self, op, pats, b, specimen, seed, snr, prior):
        coder, T, rp = self.op_info(op)
        gt = self.lab.gt[specimen]
        y = self.bin_y(self.lab._Yclean(op, specimen, "fine"), pats, rp, b)
        g = torch.Generator(device="cpu").manual_seed(seed)
        y = y + float(y.mean()) * 10 ** (-snr / 10) * torch.randn(y.shape, generator=g).numpy()
        A = snp.array(self.binned(op, pats, b))
        t = time.time()
        est = np.asarray(run_pnp(A, snp.array(y), obj_shape=gt.shape, gt=snp.array(gt), **prior)[0])
        m = score(gt, est); m["secs"] = round(time.time() - t, 1)
        self.save_volume(est, f"{op}_bin{b}", len(pats), specimen, seed, snr, pats)
        if specimen == "beads":
            m["ghost"] = float(np.mean([b_["ghost_ratio"] for b_ in bead_report(est, BEADS, mirror_plane)]))
        return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.integer, np.floating))}

    def binning(self):
        """For each measurement budget M and detector side p (bin b = 12/p), use k = M / p^2 frames."""
        op = self.args.ops[0]; coder, T, rp = self.op_info(op); nd = int(round(rp ** 0.5))
        cfg = get_config(); d = angle_scan_directions(cfg.n_alphas, cfg.n_thetas).numpy()
        configs = []
        for M in self.args.budgets:
            for p in self.args.det_sides:
                if nd % p or M % (p * p):
                    continue
                k = M // (p * p)
                if 1 <= k <= T:
                    configs.append((M, p, nd // p, k))
        print("[binning] configs (M, detector p x p, bin b, frames k): " + ", ".join(map(str, configs)), flush=True)
        def subsets_for(k):
            if k >= T:
                return {"all": [list(range(T))]}
            return {"random": [rule_random(k, T, i) for i in range(self.args.draws)], "spread": [rule_spread(k, d)]}
        snr_of = lambda M: self.args.snr + 5 * math.log10(T * rp / M)
        # 1. tune the prior scale per (M, p, preset) on the tuning set, with random draw 0 (or the full scan)
        tkey = lambda r: (r["preset"], r["M"], r["p"], r["l1s"], r["rhos"], r["max_iter"])
        have = self.done("bin_tune.jsonl", tkey)
        for (M, p, b, k) in configs:
            pats = next(iter(subsets_for(k).values()))[0]
            for preset, (spec, seed) in TUNE_ON.items():
                if preset not in {preset_of(s_) for s_ in self.args.specimens}:
                    continue
                for st in self.combos():
                    if (preset, M, p, st["l1s"], st["rhos"], st["max_iter"]) in have:
                        continue
                    m = self.solve_binned(op, pats, b, spec, seed, snr_of(M), self.prior(preset, st))
                    self.append("bin_tune.jsonl", {"op": op, "preset": preset, "M": M, "p": p, "k": k, **st, **m})
                    print(f"[bin-tune] {preset} M={M} {p}x{p} k={k} l1x{st['l1s']:g} rhox{st['rhos']:g} "
                          f"it={st['max_iter']} PSNR {m['psnr']:.2f} ({m['secs']}s)", flush=True)
        best = {}
        for r in self.lines("bin_tune.jsonl"):
            kk = (r["preset"], r["M"], r["p"])
            if kk not in best or r["psnr"] > best[kk]["psnr"]:
                best[kk] = r
        # 2. the test grid, seed-major
        key = lambda r: (r["M"], r["p"], r["rule"], r["draw"], r["specimen"], r["seed"])
        have = self.done("binning.jsonl", key)
        for seed in range(self.args.seeds):
            for (M, p, b, k) in configs:
                for rule, draws in subsets_for(k).items():
                    for draw, pats in enumerate(draws):
                        for spec in self.args.specimens:
                            if (M, p, rule, draw, spec, seed) in have:
                                continue
                            bt = {kk: best.get((preset_of(spec), M, p), self.DEFAULT)[kk] for kk in self.DEFAULT}
                            m = self.solve_binned(op, pats, b, spec, seed, snr_of(M), self.prior(preset_of(spec), bt))
                            self.append("binning.jsonl", {"op": op, "M": M, "p": p, "bin": b, "k": k, "rule": rule,
                                                          "draw": draw, "specimen": spec, "seed": seed,
                                                          "snr": snr_of(M), **bt, **m})
                            print(f"[binning] s{seed} M={M:5d} {p:2d}x{p:<2d} k={k:3d} {rule:6s}#{draw} {spec:11s} "
                                  f"PSNR {m['psnr']:.2f} depthErr {m['local_depth_err']:.2f}"
                                  + (f" ghost {m['ghost']:.2f}" if "ghost" in m else "") + f" ({m['secs']}s)", flush=True)
        self.binning_report()

    def binning_report(self):
        recs = self.lines("binning.jsonl")
        if not recs:
            return
        lines = ["# Pixels vs frames at a fixed measurement budget", "",
                 "Each row: M measurements = k frames x p^2 detector pixels, equal total dose (equal per-measurement SNR).",
                 "Cells: PSNR mean ± std (dB) / local depth error (planes), over seeds and draws.", ""]
        for spec in sorted({r["specimen"] for r in recs}):
            lines += [f"## {spec}", "", "| M | detector | frames k | random | spread / all |", "|---|---|---|---|---|"]
            for (M, p) in sorted({(r["M"], r["p"]) for r in recs}, key=lambda t: (t[0], -t[1])):
                cell = {}
                for rule in ("random", "spread", "all"):
                    v = [r for r in recs if r["specimen"] == spec and r["M"] == M and r["p"] == p and r["rule"] == rule]
                    if v:
                        ps = np.array([x["psnr"] for x in v]); es = np.array([x["local_depth_err"] for x in v])
                        cell[rule] = f"{ps.mean():.2f} ± {ps.std():.2f} / {es.mean():.2f}"
                k = next(r["k"] for r in recs if r["M"] == M and r["p"] == p)
                lines.append(f"| {M} | {p}x{p} | {k} | {cell.get('random', '')} | {cell.get('spread', cell.get('all', ''))} |")
            lines.append("")
        (self.out / "binning.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines)); print(f"[binning] -> {self.out / 'binning.md'}")

    # --- report ---
    def report(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        recs = self.lines("runs.jsonl")
        groups = {}
        for r in recs:
            groups.setdefault((r["op"], r["specimen"], r["dose"], r["k"], r["rule"]), []).append(r)
        # the full scan serves both dose regimes
        for (op, spec, dose, k, rule), v in list(groups.items()):
            if rule == "all":
                for d in ("frame", "total"):
                    groups.setdefault((op, spec, d, k, rule), v)
        lines = ["# Compression: reconstruction from k of 256 patterns", "",
                 "Measurements M = 144 k against 36,864 voxels: " + ", ".join(
                     f"k={k}: M={144 * k} ({144 * k / 36864:.3f} of the voxels, {256 // k}x fewer frames)" for k in KS[:-1]) + ".", "",
                 "Tuning: L1 weight and ADMM penalty per k on blood_cell (DnCNN prior) / beads seed 100 (L1 prior); never on test data.", "",
                 f"SNR {self.args.snr:g} dB per frame ('frame') or the 256-frame photon budget shared ('total').",
                 "mean ± std over noise seeds (and random draws). Pass bar for a selection rule: ≥ 1 dB over random at k = 8 or 16.", ""]
        ops = sorted({g[0] for g in groups}); specs = sorted({g[1] for g in groups}); doses = sorted({g[2] for g in groups})
        fig, axes = plt.subplots(2, len(specs) * len(doses), figsize=(4.2 * len(specs) * len(doses), 7), squeeze=False)
        col = 0
        for spec in specs:
            for dose in doses:
                lines += [f"## {spec}, dose = {dose}", "", "| op | rule | " + " | ".join(f"k={k}" for k in KS) + " |",
                          "|---|---|" + "---|" * len(KS)]
                for op in ops:
                    rules = sorted({g[4] for g in groups if g[0] == op and g[1] == spec and g[2] == dose})
                    if len(rules) > 1:
                        rules.remove("all") if "all" in rules else None
                    full = groups.get((op, spec, dose, max(KS), "all"))
                    for rule in rules:
                        cells, xs, ps, pe, ds, de = [], [], [], [], [], []
                        for k in KS:
                            v = groups.get((op, spec, dose, k, rule)) or groups.get((op, spec, dose, k, "all"))
                            if not v:
                                cells.append(""); continue
                            p = np.array([x["psnr"] for x in v]); e = np.array([x["local_depth_err"] for x in v])
                            cells.append(f"{p.mean():.2f} ± {p.std():.2f} / {e.mean():.2f}")
                            xs.append(k); ps.append(p.mean()); pe.append(p.std()); ds.append(e.mean()); de.append(e.std())
                        lines.append(f"| {op} | {rule} | " + " | ".join(cells) + " |")
                        lab = f"{op}:{rule}"
                        axes[0, col].errorbar(xs, ps, yerr=pe, marker="o", capsize=3, label=lab)
                        axes[1, col].errorbar(xs, ds, yerr=de, marker="o", capsize=3, label=lab)
                    if full:
                        ref = np.mean([x["psnr"] for x in full])
                        axes[0, col].axhline(ref - 1, ls=":", c="gray")
                lines += ["", "cells: PSNR mean ± std (dB) / local depth error (planes)", ""]
                for row, yl in ((0, "PSNR (dB)"), (1, "local depth error (planes)")):
                    ax = axes[row, col]; ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
                    ax.set_xlabel("patterns k (of 256)"); ax.set_ylabel(yl); ax.set_title(f"{spec}, dose={dose}")
                    ax.grid(alpha=0.3)
                col += 1
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(self.out / "compression.png", dpi=150)
        (self.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines)); print(f"[report] -> {self.out / 'report.md'}, {self.out / 'compression.png'}")

    # --- smoke ---
    def smoke(self):
        """Every stage end to end on tiny settings, written to <out>/smoke."""
        a = self.args
        a.out = str(Path(a.out) / "smoke"); self.out = _ROOT / a.out; self.out.mkdir(parents=True, exist_ok=True)
        a.ks, a.draws, a.l1s, a.rhos, a.iters, a.n_val, a.seeds = [8, 16], 1, [1.0], [1.0], [2], 3, 1
        a.pred_probes, a.pred_iters = 2, 5
        a.doses, a.specimens = ["frame", "total"], ["neural_cell", "beads"]
        self.select(); self.tune(); self.validate(); self.run(); self.report()
        a.budgets, a.det_sides = [1152], [12, 6]
        self.binning()


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruction from k of the scan patterns, by subset rule.")
    ap.add_argument("stage", choices=["select", "tune", "validate", "run", "report", "binning", "smoke"])
    ap.add_argument("--ops", default="random_gap66", help="comma-separated registered operators")
    ap.add_argument("--out", default="results/recon/compression")
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--draws", type=int, default=5, help="random-rule draws")
    ap.add_argument("--specimens", default=",".join(TEST))
    ap.add_argument("--seeds", type=int, default=3, help="noise seeds 0..n-1")
    ap.add_argument("--doses", default="frame,total")
    ap.add_argument("--snr", type=float, default=25.0, help="per-frame SNR (dB) at k = 256")
    ap.add_argument("--l1s", default="0.5,1,2", help="tune: L1-weight scales")
    ap.add_argument("--rhos", default="0.5,1,2", help="tune: ADMM penalty scales (rho and l1_rho)")
    ap.add_argument("--iters", default="20", help="tune: ADMM iteration counts")
    ap.add_argument("--tune_presets", default="", help="tune: only these presets (comma list; default all)")
    ap.add_argument("--tune_dose", default="total", choices=["frame", "total"],
                    help="tune at the SNR of this dose regime (the main compression claim uses 'total')")
    ap.add_argument("--val_k", type=int, default=16)
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--full_pred", type=int, default=1, help="validate: also the full-operator Bayesian predictors")
    ap.add_argument("--betas", default="0,10", help="validate: lateral-smoothness weights of the full predictor")
    ap.add_argument("--pred_probes", type=int, default=8, help="validate: Hutchinson probes of the full predictor")
    ap.add_argument("--pred_iters", type=int, default=60, help="validate: CG iterations of the full predictor")
    ap.add_argument("--t_ref", type=int, default=256, help="frames of the full scan (photon budget of dose 'total')")
    ap.add_argument("--budgets", default="1152,2304,4608", help="binning: measurement budgets M")
    ap.add_argument("--det_sides", default="12,6,4,3", help="binning: detector sides p (must divide 12)")
    ap.add_argument("--worker", type=int, default=0, help="run stage: this worker's index")
    ap.add_argument("--n_workers", type=int, default=1, help="run stage: split the grid over N processes")
    a = ap.parse_args()
    a.ops = a.ops.split(","); a.ks = [int(x) for x in a.ks.split(",")]; a.rules = a.rules.split(",")
    a.specimens = a.specimens.split(","); a.doses = a.doses.split(",")
    a.betas = [float(x) for x in a.betas.split(",")]
    a.budgets = [int(x) for x in a.budgets.split(",")]; a.det_sides = [int(x) for x in a.det_sides.split(",")]
    a.l1s = [float(x) for x in a.l1s.split(",")]; a.rhos = [float(x) for x in a.rhos.split(",")]
    a.iters = [int(x) for x in a.iters.split(",")]
    a.tune_presets = [x for x in a.tune_presets.split(",") if x]
    r = Runner(a)
    getattr(r, a.stage)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
