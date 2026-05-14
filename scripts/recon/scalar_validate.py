#!/usr/bin/env python3
"""
Validate the scalar two-layer coder and the shared forward operator on neural_cell.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import scico.numpy as snp

from configs.default import get_config
from coder.scalar import build_scalar_coder
from pipeline.forward import ExportForward, emission_psf, add_noise_snr
from pipeline.reconstruct import run_pnp, PRIOR_PRESETS as PRESETS
from pipeline.metrics import axial_profile, depth_metrics
from pipeline import viz


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate scalar coder reaches ~35 dB.")
    ap.add_argument("--xr", default="backup_fyp/D2NN/Xr_neural_cell.pt")
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--d_size", type=int, default=16)
    ap.add_argument("--patterns", type=int, default=128)
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--compressions", default="1,8")
    ap.add_argument("--spread_chunk", type=int, default=32, help="patterns per window-8 H2 FFT batch")
    ap.add_argument("--fwd_chunk", type=int, default=0, help="patterns per build_A conv batch (0=all; keep 0)")
    ap.add_argument("--pattern_batch", type=int, default=64,
                    help="build A in row-batches of this many patterns (memory-safe; concatenated)")
    ap.add_argument("--h2", default=None, help="load a designed coder's speckle_volume.pt instead of a fresh random coder")
    ap.add_argument("--save_a", default=None, help="optional path to torch.save the built A")
    ap.add_argument("--out", default="results/recon/scalar_validate")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(); cfg.device = args.device; dev = args.device
    torch.manual_seed(args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S"); out = Path(args.out) / stamp; out.mkdir(parents=True, exist_ok=True)

    if args.h2:                                                     # load a designed coder's H2
        H2 = torch.load(args.h2, map_location=dev).float()[:args.patterns].contiguous()
        print(f"loaded H2 from {args.h2}")
    else:
        coder = build_scalar_coder(cfg)
        with torch.no_grad():
            H2 = coder(chunk=args.spread_chunk)[:args.patterns].contiguous()   # (T, nz, N, N)
        del coder; torch.cuda.empty_cache()
    nz, N = cfg.nz, cfg.n_scalar
    emPSF = emission_psf(nz, N, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
    torch.cuda.empty_cache()
    # Build A in row-batches of `pattern_batch` patterns (each a memory-safe non-chunked build),
    # concatenated pattern-major -> the full (T*Nd*Nd, n_vox) operator.
    pb = args.pattern_batch or args.patterns
    t = time.time(); parts = []
    for s in range(0, args.patterns, pb):
        fwd = ExportForward(H2[s:s + pb], emPSF, patch=(4, 4, 4), d_size=args.d_size,
                            chunk=args.fwd_chunk, dose_mean=cfg.dose_mean, device=dev)
        ts = time.time(); parts.append(fwd.build_A().cpu())
        print(f"  patterns {s}:{s + pb} -> A rows {parts[-1].shape[0]} ({time.time()-ts:.0f}s)")
        del fwd; torch.cuda.empty_cache()
    A = torch.cat(parts, 0).to(dev)
    print(f"scalar A {tuple(A.shape)} built {time.time()-t:.0f}s")
    if args.save_a:
        torch.save(A.detach().cpu(), args.save_a)

    Xr = torch.load(args.xr, map_location=dev)
    gt = Xr[args.point].reshape(8, 32, 32).float()
    gt_np = gt.detach().cpu().numpy(); gt_j = snp.array(gt_np); gt_std = float(axial_profile(gt_np).std())
    Y = add_noise_snr(A @ gt.reshape(-1), args.snr, seed=args.seed)
    A_j, Y_j = snp.array(A.detach().cpu().numpy()), snp.array(Y.detach().cpu().numpy())
    n_vox = 8192
    print(f"point={args.point} d_size={args.d_size} patterns={args.patterns} GT axial-std={gt_std:.4f}")

    for c in [float(x) for x in args.compressions.split(",")]:
        nm = min(A.shape[0], int(round(n_vox / c)))
        vols = {"GT": gt_np}; profs = {}
        for pname, prior in PRESETS.items():
            est, _ = run_pnp(A_j, Y_j, obj_shape=(8, 32, 32), gt=gt_j, n_measure=nm, **prior)
            est = np.asarray(est); m = depth_metrics(gt_np, est)
            sh = float(axial_profile(est).std() / (gt_std + 1e-12))
            print(f"  c={c:g} nm={nm:5d}  {pname:6s}: psnr={m['psnr']:.2f}dB  "
                  f"centroid_err={m['centroid_err']:.3f}  profile_l1={m['profile_l1']:.4f}  shape={sh:.2f}")
            vols[pname] = est; profs[pname] = axial_profile(est)
        viz.show_mips(vols, title=f"scalar D2NN (seed {args.seed}) c={c:g}").savefig(
            out / f"mips_c{c:g}.png", dpi=140, bbox_inches="tight")
        viz.overlay_axial_profile(axial_profile(gt_np), profs, title=f"scalar D2NN c={c:g}").savefig(
            out / f"profile_c{c:g}.png", dpi=140, bbox_inches="tight")
    print(f"artifacts in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
