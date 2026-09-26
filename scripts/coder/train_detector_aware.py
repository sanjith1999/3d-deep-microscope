#!/usr/bin/env python3
"""
Detector-aware training of the single phase plate.

Minimises the squared correlation between the operator columns of voxels stacked in depth, computed through the
real detector (emission NA, pixel pooling, chosen frames), with mirror pairs about the focus weighted twice.

    python -u scripts/coder/train_detector_aware.py --na_em 0.1 --d 8 --frames patterns_d8.json --name da_na01_d8
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import torch
import torch.nn.functional as F

from configs.default import get_config
from coder import baselines
from coder.psf import conv_3D
from pipeline.forward import emission_psf


def footprint_table(emPSF, nz, N, d, dev):
    m = nz // 2; c = N // 2
    X = torch.zeros(nz, nz, N, N, device=dev)
    for z in range(nz):
        X[z, z, c, c] = 1.0
    K = conv_3D(emPSF, X.to(torch.complex64), 2).real[:, m]
    E = torch.zeros(nz, 3 * N + d, 3 * N + d, device=dev); E[:, N:2 * N, N:2 * N] = K
    return F.avg_pool2d(E.unsqueeze(1), d, stride=1).squeeze(1)


def gather_g(S, z, x, y, N, d):
    Nd = N // d; i = torch.arange(Nd, device=S.device); c = N // 2
    ax = d * i[None, :] - x[:, None] + c + N; ay = d * i[None, :] - y[:, None] + c + N
    return S[z[:, None, None], ax[:, :, None], ay[:, None, :]]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--na_em", type=float, required=True); ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--frames", required=True, help="json list of scan-pattern indices (the frames this plate is trained for)")
    ap.add_argument("--name", required=True); ap.add_argument("--iters", type=int, default=600); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch_xy", type=int, default=24); ap.add_argument("--frames_per_step", type=int, default=8)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed)
    cfg = get_config(); cfg.device = dev; cfg.n_layers = 1; cfg.na_em = a.na_em
    frames = json.loads((_ROOT / a.frames).read_text()) if not Path(a.frames).is_absolute() else json.loads(Path(a.frames).read_text())
    model = baselines.d2nn_model(cfg, seed=0)                         # the random plate used everywhere (init)
    for p in model.phases:
        p.requires_grad_(True)
    nz, N = cfg.nz, cfg.n_fine // cfg.bin_factor
    emPSF = emission_psf(nz, N, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
    S = footprint_table(emPSF, nz, N, a.d, dev); del emPSF; torch.cuda.empty_cache()
    rz, rx = nz // 2, N // 2
    mirror = torch.zeros(rz, rz, device=dev)
    for zc in range(rz):
        if 0 <= 14 - zc < rz and 14 - zc != zc:
            mirror[zc, 14 - zc] = 1.0
    W = (1 - torch.eye(rz, device=dev)) * (1 + mirror)                # off-diagonal, mirror pairs x2
    opt = torch.optim.Adam(model.phases, lr=a.lr); hist = []; t0 = time.time()

    def loss_for(fr_idx, xy):
        H = model(torch.tensor(fr_idx, device=dev))                   # (T, nz, N, N) illumination (unnormalised; scale-free loss)
        cols = []
        for (xc, yc) in xy:
            zs, xs, ys = [], [], []
            for zc in range(rz):
                for dz in (0, 1):
                    for dx in (0, 1):
                        for dy in (0, 1):
                            zs.append(2 * zc + dz); xs.append(2 * xc + dx); ys.append(2 * yc + dy)
            z = torch.tensor(zs, device=dev); x = torch.tensor(xs, device=dev); y = torch.tensor(ys, device=dev)
            g = gather_g(S, z, x, y, N, a.d)                          # (rz*8, Nd, Nd)
            Iv = H[:, z, x, y]                                        # (T, rz*8)
            m = (Iv[:, :, None, None] * g[None]).reshape(len(fr_idx), rz, 8, -1).sum(2)   # (T, rz, Nd^2): patch sums
            cols.append(m.permute(1, 0, 2).reshape(rz, -1))           # (rz, T * Nd^2)
        C = torch.stack(cols)                                         # (B, rz, M)
        C = C - C.mean(-1, keepdim=True); C = C / (C.norm(dim=-1, keepdim=True) + 1e-12)
        G = C @ C.transpose(1, 2)                                     # (B, rz, rz) depth-correlation matrices
        return ((G ** 2) * W).sum((1, 2)).mean() / W.sum(), G

    for it in range(1, a.iters + 1):
        fr = sorted(torch.randperm(len(frames))[:min(a.frames_per_step, len(frames))].tolist())
        xy = [(int(torch.randint(2, rx - 2, (1,))), int(torch.randint(2, rx - 2, (1,)))) for _ in range(a.batch_xy)]
        loss, G = loss_for([frames[i] for i in fr], xy)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 25 == 0 or it == 1:
            off = G.detach().abs() * (1 - torch.eye(rz, device=dev))
            hist.append({"it": it, "loss": float(loss), "mean_abs_depth_corr": float(off.sum() / (off.shape[0] * rz * (rz - 1))),
                         "max_abs_depth_corr": float(off.amax())})
            print(f"[da] {a.name} it {it} loss {float(loss):.4f} mean|corr| {hist[-1]['mean_abs_depth_corr']:.3f} "
                  f"max {hist[-1]['max_abs_depth_corr']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    out = _ROOT / "results/designs"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"phases": [p.detach().cpu() for p in model.phases],
                "meta": {"objective": "detector-aware depth decorrelation of operator columns", "na_em": a.na_em, "d": a.d,
                         "frames": frames, "init": "random plate seed 0 (random1L)", "iters": a.iters, "lr": a.lr}, "hist": hist},
               out / f"{a.name}.pt")
    print(f"[da] saved {out / (a.name + '.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
