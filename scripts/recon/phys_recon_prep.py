#!/usr/bin/env python3
"""
Physical-grid reconstruction test, stage 1: clean measurements and ground truth.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from configs.default import get_config
from coder import baselines
from pipeline.forward import ExportForward, emission_psf, matched_dose, reduce_object, bead_phantom
from pipeline.specimens import BEADS, stored_specimen as specimen


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean measurements + GT on the physical grid.")
    ap.add_argument("--coder_ckpt", required=True)
    ap.add_argument("--specimen", default="backup_fyp/D2NN/X_neural_cell.pt")
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--d_size", type=int, default=8)
    ap.add_argument("--out", default="results/recon/phys_recon_inputs.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(); cfg.device = dev = args.device
    T, N, nz = cfg.n_alphas * cfg.n_thetas, cfg.n_fine // cfg.bin_factor, cfg.nz
    patch = tuple(cfg.recon_patch)

    H2 = matched_dose(baselines.designed_volume(cfg, args.coder_ckpt)[:T], cfg.dose_mean)
    torch.cuda.empty_cache()
    emPSF = emission_psf(nz, N, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
    fwd = ExportForward(H2, emPSF, patch=patch, d_size=args.d_size, chunk=16, dose_mean=None, device=dev)

    objs = {"neural_cell": specimen(args.specimen, args.point, nz, N),
            "beads": bead_phantom(nz, N, BEADS, patch=patch, radius_um=0.8,
                                  dz_um=cfg.dz_um, pitch_um=cfg.pitch_scalar, device="cpu")}
    out = {"beads_list": BEADS, "patch": patch}
    for name, X in objs.items():
        X = X.to(dev)
        Y = fwd.forward_fine(X).reshape(-1).float().cpu()
        gt = reduce_object(X, patch).float().cpu()
        out[name] = {"Y": Y, "gt": gt}
        print(f"[{name}] X {tuple(X.shape)} -> Y {tuple(Y.shape)}  GT {tuple(gt.shape)} "
              f"GT max {gt.max():.3f} nonzero {(gt > 0.05).float().mean():.3f}", flush=True)
    torch.save(out, args.out)
    print(f"saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
