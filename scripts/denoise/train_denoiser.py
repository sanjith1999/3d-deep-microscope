#!/usr/bin/env python3
"""
Train a domain denoiser for the PnP solver on screened crops prepared exactly like the test references.
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import torch
import torch.nn.functional as F
from pipeline.specimens import STORED
from pipeline.learned_denoiser import DnCNN2D, DnCNN3D, UNet3D


def recon_volume(X, i):
    x = X[i].reshape(32, 128, 128).float()
    x = F.adaptive_avg_pool3d(x[None, None], (32, 96, 96))[0, 0]
    x = x / (x.max() + 1e-12)
    return F.avg_pool3d(x[None, None], 2)[0, 0]                      # (16, 48, 48)


def augment(v):
    for d in (0, 1, 2):
        if random.random() < 0.5:
            v = v.flip(d)
    if random.random() < 0.5:
        v = v.transpose(1, 2)
    return v * random.uniform(0.3, 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stack", default="neural_cell"); ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--name", required=True); ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None); ap.add_argument("--sigma_max", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--n_val", type=int, default=16)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="dncnn", choices=["dncnn", "unet3d"])
    ap.add_argument("--nonblind", action="store_true", help="sigma map as a second input channel (for annealed PnP-HQS)")
    a = ap.parse_args(); dev = a.device; random.seed(a.seed); torch.manual_seed(a.seed)
    iters = a.iters or (30000 if a.dim == 2 else 15000); batch = a.batch or (64 if a.dim == 2 else 8)
    ov = json.loads((_ROOT / f"results/denoisers/overlap_{a.stack}.json").read_text())
    ids = list(ov["train"]); random.shuffle(ids); val_ids, tr_ids = ids[:a.n_val], ids[a.n_val:]
    X = torch.load(_ROOT / STORED[a.stack], map_location="cpu", mmap=True)
    t0 = time.time()
    V = torch.stack([recon_volume(X, i) for i in tr_ids]).to(dev)           # (n, 16, 48, 48)
    Vv = torch.stack([recon_volume(X, i) for i in val_ids]).to(dev)
    print(f"[train] {a.name}: {len(tr_ids)} train / {len(val_ids)} held-out volumes from {a.stack} ({time.time() - t0:.0f}s)", flush=True)
    ic = 2 if a.nonblind else 1
    if a.arch == "unet3d":
        a.dim = 3; model = UNet3D(width=32, in_ch=ic).to(dev)
    else:
        model = (DnCNN2D(in_ch=ic) if a.dim == 2 else DnCNN3D(in_ch=ic)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=iters, pct_start=0.05)
    g = torch.Generator(device=dev).manual_seed(a.seed + 1)

    def batch_2d(src, n):
        o = random.choice(("xy", "xz", "yz"))
        vols = [augment(src[random.randrange(len(src))]) for _ in range(n)]
        if o == "xy":
            ims = [v[random.randrange(16)] for v in vols]
        elif o == "xz":
            ims = [v[:, :, random.randrange(48)] for v in vols]
        else:
            ims = [v[:, random.randrange(48), :] for v in vols]
        return torch.stack(ims)[:, None]

    def batch_3d(src, n):
        return torch.stack([augment(src[random.randrange(len(src))]) for _ in range(n)])[:, None]

    mk = batch_2d if a.dim == 2 else batch_3d
    hist = []
    for it in range(1, iters + 1):
        model.train(); x = mk(V, batch)
        sig = torch.rand(x.shape[0], *[1] * (x.dim() - 1), device=dev, generator=g) * a.sigma_max
        y = x + sig * torch.randn(x.shape, device=dev, generator=g)
        loss = F.mse_loss(model(y, sig.flatten()) if a.nonblind else model(y), x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0 or it == iters:
            model.eval(); res = {}
            with torch.no_grad():
                gv = torch.Generator(device=dev).manual_seed(123)
                for s in (0.05, 0.1, 0.2):
                    xv = Vv[:, None] if a.dim == 3 else Vv.reshape(-1, 1, 48, 48)
                    yv = xv + s * torch.randn(xv.shape, device=dev, generator=gv)
                    mse = F.mse_loss(model(yv, s) if a.nonblind else model(yv), xv).item(); mse_in = F.mse_loss(yv, xv).item()
                    res[f"psnr_gain_sigma{s}"] = 10 * math.log10(mse_in / mse)
            hist.append({"it": it, "loss": loss.item(), **res})
            print(f"[train] {a.name} it {it} loss {loss.item():.2e} held-out PSNR gain (dB) "
                  + " ".join(f"{k[-4:]}:{v:.2f}" for k, v in res.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    out = _ROOT / "results/denoisers"; out.mkdir(parents=True, exist_ok=True)
    meta = {"stack": a.stack, "dim": a.dim, "train_crops": len(tr_ids), "heldout_crops": val_ids, "iters": iters, "batch": batch,
            "sigma_max": a.sigma_max, "lr": a.lr, "grid": "recon 16x48x48 (0.64 um)", "overlap_screen": f"overlap_{a.stack}.json",
            "secs": round(time.time() - t0)}
    d = 2 if a.dim == 2 else 3
    meta["nonblind"] = bool(a.nonblind)
    torch.save({"state": model.state_dict(), "dim": d, "depth": 17 if d == 2 else 12, "width": 64 if d == 2 else 48,
                "in_ch": ic, "arch": a.arch, "meta": meta} if a.arch == "dncnn" else
               {"state": model.state_dict(), "dim": 3, "arch": "unet3d", "width": 32, "in_ch": ic, "meta": meta},
               out / f"{a.name}.pt")
    (out / f"{a.name}.json").write_text(json.dumps({"meta": meta, "history": hist}, indent=1))
    print(f"[train] saved {out / (a.name + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
