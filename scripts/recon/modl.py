#!/usr/bin/env python3
"""
MoDL (Aggarwal et al. 2019): unrolled CG data consistency with a shared 3D CNN, trained end to end
through the real one-plate operator.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, random, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon", _ROOT / "scripts" / "denoise", _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import numpy as np
import torch
import torch.nn as nn

from pipeline.learned_denoiser import DnCNN3D

SHAPE = (16, 48, 48); N_VOX = 16 * 48 * 48; RP = 144; T = 256
COHORT = [f"neural_cell_p{i}" for i in (64, 128, 192, 320, 448, 576, 704, 832)]
VAL = ["neural_cell_p303", "neural_cell_p10", "neural_cell_p693"]


def cg(op, b, x, iters):
    """Batched CG on columns of b (N_VOX, B); differentiable (autograd through the iterations)."""
    r = b - op(x); p = r; rs = (r * r).sum(0, keepdim=True)
    for _ in range(iters):
        Ap = op(p); a = rs / ((p * Ap).sum(0, keepdim=True) + 1e-30)
        x = x + a * p; r = r - a * Ap; rn = (r * r).sum(0, keepdim=True)
        p = r + (rn / (rs + 1e-30)) * p; rs = rn
    return x


class MoDL(nn.Module):
    def __init__(self, n_unroll=6, cg_iters=10, depth=8, width=32, out_z=False):
        super().__init__()
        self.D = DnCNN3D(depth=depth, width=width); self.log_mu = nn.Parameter(torch.tensor(math.log(0.05)))
        self.n, self.cgi, self.out_z = n_unroll, cg_iters, out_z

    def forward(self, A, y, L, trace=None):
        """A: (M, N_VOX) rows used; y: (M, B)."""
        mu = torch.exp(self.log_mu) * L
        op = lambda v: A.t() @ (A @ v) + mu * v
        Aty = A.t() @ y
        x = cg(op, Aty, torch.zeros_like(Aty), self.cgi)
        for i in range(self.n):
            B = x.shape[1]
            z = self.D(x.t().reshape(B, 1, *SHAPE)).reshape(B, N_VOX).t()
            if self.out_z and i == self.n - 1:              # end on the denoiser (no final noise re-injection)
                x = z
            else:
                x = cg(op, Aty + mu * z, x, self.cgi)
            if trace is not None:
                trace.append(x.detach().t().reshape(B, *SHAPE).clamp(0, 1).cpu().numpy())
        return x.t().reshape(-1, *SHAPE)


def lam_max(A, it=15):
    v = torch.randn(A.shape[1], 1, device=A.device)
    for _ in range(it):
        v = A.t() @ (A @ v); v = v / v.norm()
    return float((v * (A.t() @ (A @ v))).sum())


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["train", "test"])
    ap.add_argument("--name", default="modl_a"); ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--unroll", type=int, default=6); ap.add_argument("--cg", type=int, default=8)
    ap.add_argument("--ks", default="32,64,256"); ap.add_argument("--q", type=float, default=15)
    ap.add_argument("--depth", type=int, default=8); ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--init", default=None, help="checkpoint of a blind 3D denoiser to initialise D (e.g. nc3d)")
    ap.add_argument("--out_z", action="store_true")
    ap.add_argument("--op", default="random1L_gap66"); ap.add_argument("--dir", default="results/recon/compression_1L_q15")
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0); random.seed(0)
    out = _ROOT / "results/paper/modl" / a.name; out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((_ROOT / "results/recon/ops" / a.op / "meta.json").read_text())
    A_full = torch.load(_ROOT / meta["A"], map_location="cpu", mmap=True).float().to(dev)
    model = MoDL(a.unroll, a.cg, a.depth, a.width, a.out_z).to(dev)
    if a.init and a.stage == "train":
        ck0 = torch.load(_ROOT / "results/denoisers" / f"{a.init}.pt", map_location=dev); model.D.load_state_dict(ck0["state"])
        print(f"[modl] D initialised from {a.init}", flush=True)
    L256 = lam_max(A_full); print(f"[modl] ||A||^2 (256 frames) = {L256:.4g}", flush=True)
    rows_of = lambda pats: (np.asarray(pats)[:, None] * RP + np.arange(RP)[None, :]).reshape(-1)
    sys.path.insert(0, str(_ROOT / "scripts" / "recon"))
    from compression_subsets import Runner
    r = Runner(argparse.Namespace(out=a.dir, snr=a.q, t_ref=256))
    subs = json.loads((_ROOT / a.dir / "subsets.json").read_text())[a.op]

    def rowsA(k, pats=None):
        if k >= T:
            return A_full, L256
        pats = pats if pats is not None else sorted(random.sample(range(T), k))
        idx = torch.from_numpy(rows_of(pats)).to(dev); Ak = A_full[idx]
        return Ak, L256 * k / T                                     # E||A_k||^2 ~ (k/T) ||A||^2 (normalisation only)

    if a.stage == "train":
        from train_denoiser import recon_volume
        from pipeline.specimens import STORED
        ov = json.loads((_ROOT / "results/denoisers/overlap_neural_cell.json").read_text())
        held = set(json.loads((_ROOT / "results/denoisers/nc2d.json").read_text())["meta"]["heldout_crops"])
        ids = [i for i in ov["train"] if i not in held]
        X = torch.load(_ROOT / STORED["neural_cell"], map_location="cpu", mmap=True)
        V = torch.stack([recon_volume(X, i) for i in ids]).reshape(len(ids), N_VOX).to(dev)        # (n, N_VOX)
        print(f"[modl] {len(ids)} training volumes", flush=True)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        val = [(np.asarray(r.lab.gt[s]), s) for s in VAL]; hist = []; t0 = time.time()
        for it in range(1, a.iters + 1):
            model.train(); k = random.choices([16, 32, 64, 256], weights=[3, 3, 3, 1])[0]; Ak, Lk = rowsA(k)
            b = torch.randint(0, len(ids), (a.batch,), device=dev); xg = V[b].t()                 # (N_VOX, B)
            flip = [d for d in (1, 2, 3) if random.random() < 0.5]                                # lateral/axial flips
            if flip:
                xg = xg.t().reshape(-1, *SHAPE).flip(flip).reshape(-1, N_VOX).t()
            xg = xg * torch.empty(1, a.batch, device=dev).uniform_(0.5, 1.0)
            y0 = Ak @ xg
            sig = (A_full @ xg).mean(0, keepdim=True) * 10 ** (-(a.q + 5 * math.log10(T / k)) / 10)
            y = y0 + sig * torch.randn_like(y0)
            xh = model(Ak, y, Lk)
            loss = ((xh - xg.t().reshape(-1, *SHAPE)) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if it % 250 == 0 or it == a.iters:
                model.eval(); ps = []
                with torch.no_grad():
                    for gt, s in val:
                        for kk in (32, 64):
                            Ak2, Lk2 = rowsA(kk, subs[str(kk)]["random"][0]); rows = rows_of(subs[str(kk)]["random"][0])
                            yv, _ = r.lab.measure(a.op, s, r.snr_for(kk, T, "total"), 100)
                            yv = torch.from_numpy(np.asarray(yv)[rows]).float().to(dev)[:, None]
                            est = model(Ak2, yv, Lk2)[0].clamp(0, 1).cpu().numpy()
                            ps.append(10 * math.log10((gt.max() - gt.min()) ** 2 / np.mean((est - gt) ** 2)))
                hist.append({"it": it, "loss": loss.item(), "val_psnr_k32_k64": float(np.mean(ps)), "mu": float(torch.exp(model.log_mu))})
                print(f"[modl] it {it} loss {loss.item():.2e} val PSNR (nc held-out, fine physics, k32/64) {np.mean(ps):.2f} "
                      f"mu {float(torch.exp(model.log_mu)):.3g} ({time.time() - t0:.0f}s)", flush=True)
                torch.save({"state": model.state_dict(), "args": vars(a), "hist": hist}, out / "model.pt")
        (out / "train.json").write_text(json.dumps({"args": vars(a), "n_train": len(ids), "hist": hist}, indent=1))
        return 0
    # ---- test ----
    ck = torch.load(out / "model.pt", map_location=dev); model.load_state_dict(ck["state"]); model.eval()
    runs = [json.loads(l) for f in sorted(glob.glob(str(_ROOT / a.dir / "runs*.jsonl"))) for l in open(f) if l.strip()]
    from pipeline.metrics import score
    from pipeline.metrics_paper import ssim3d_foreground
    rows, traces = [], {}
    jobs = [("neural_cell", s) for s in (0, 1, 2)] + [(c, 0) for c in COHORT]
    for k in [int(x) for x in a.ks.split(",")]:
        pats = list(range(T)) if k >= T else subs[str(k)]["random"][0]
        Ak, Lk = rowsA(k, None if k >= T else pats); rws = rows_of(pats)
        for spec, seed in jobs:
            gt = np.asarray(r.lab.gt[spec]); yv, _ = r.lab.measure(a.op, spec, r.snr_for(k, T, "total"), seed)
            yv = torch.from_numpy(np.asarray(yv)[rws]).float().to(dev)[:, None]; tr = []; t = time.time()
            with torch.no_grad():
                est = model(Ak, yv, Lk, trace=tr)[0].clamp(0, 1).cpu().numpy()
            m = score(gt, est); fg = ssim3d_foreground(gt, est)
            base = [x for x in runs if x["specimen"] == spec and x["k"] == k and x["seed"] == seed and x.get("draw", 0) == 0]
            b = base[0] if base else {}
            rows.append({"k": k, "specimen": spec, "seed": seed, "psnr": m["psnr"], "depth_um": m["local_depth_err"] * 0.64,
                         "fg_ssim": fg, "secs": round(time.time() - t, 2), "pnp_psnr": b.get("psnr"),
                         "pnp_depth_um": b.get("local_depth_err", float("nan")) * 0.64 if b else None})
            traces[f"{spec}_k{k}_s{seed}"] = [float(np.linalg.norm(e[0] - gt) / np.linalg.norm(gt)) for e in tr]
            np.savez_compressed(out / f"vol_{spec}_k{k}_s{seed}.npz", est=est.astype(np.float32), gt=gt)
            print(f"[modl test] k={k:3d} {spec:17s} s{seed}: PSNR {m['psnr']:.2f} (PnP {b.get('psnr', float('nan')):.2f})  depth "
                  f"{m['local_depth_err'] * 0.64:.3f} um (PnP {b.get('local_depth_err', float('nan')) * 0.64:.3f})  fg {fg:.3f}", flush=True)
    with open(out / "test_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "traces.json").write_text(json.dumps(traces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
