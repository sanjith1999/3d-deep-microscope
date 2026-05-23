#!/usr/bin/env python3
"""
Designed vs DMD with the vectorial excitation model and fine-object measurements.
"""
from __future__ import annotations
import argparse, json, time
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
from coder import baselines
from pipeline.forward import (ExportForward, emission_psf, add_noise_snr, reduce_object, matched_dose)
from pipeline.reconstruct import run_pnp, PRIOR_PRESETS as PRESETS
from pipeline.metrics import axial_profile, depth_metrics
from pipeline import viz


def build_A_batched(H2, emPSF, d_size, pb, fwd_chunk, dev):
    """Build A in row-batches of `pb` patterns (H2 already dose-normalised; dose_mean=None)."""
    T = H2.shape[0]; parts = []
    for s in range(0, T, pb):
        fwd = ExportForward(H2[s:s + pb], emPSF, patch=(4, 4, 4), d_size=d_size,
                            chunk=fwd_chunk, dose_mean=None, device=dev)
        ts = time.time(); parts.append(fwd.build_A().cpu())
        print(f"    patterns {s}:{s + pb} -> rows {parts[-1].shape[0]} ({time.time()-ts:.0f}s)")
        del fwd; torch.cuda.empty_cache()
    return torch.cat(parts, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Corrected designed-vs-DMD reproduction (vector model).")
    ap.add_argument("--designed", required=True, help="speckle_volume.pt of the designed vector coder")
    ap.add_argument("--xfine", default="backup_fyp/D2NN/X_neural_cell.pt")
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--d_size", type=int, default=16)
    ap.add_argument("--patterns", type=int, default=128)
    ap.add_argument("--pattern_batch", type=int, default=64)
    ap.add_argument("--fwd_chunk", type=int, default=0)
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--compressions", default="1,8")
    ap.add_argument("--out", default="results/recon/repro_vector")
    ap.add_argument("--save_a", default="results/recon")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(); cfg.device = args.device; dev = args.device
    T, Nc, nz = args.patterns, cfg.n_fine // cfg.bin_factor, cfg.nz
    emPSF = emission_psf(nz, Nc, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
    stamp = time.strftime("%Y%m%d_%H%M%S"); out = Path(args.out) / stamp; out.mkdir(parents=True, exist_ok=True)
    Path(args.save_a).mkdir(parents=True, exist_ok=True)

    # fine object -> measurement source; GT = avg-pool of the SAME object (load on CPU, move 1 point)
    raw = torch.load(args.xfine, map_location="cpu")
    Xfine = raw[args.point].reshape(nz, Nc, Nc).float().to(dev); del raw
    gt = reduce_object(Xfine).detach().cpu().numpy()                # (8,32,32) avg-pool
    gt_j = snp.array(gt); gt_std = float(axial_profile(gt).std())
    print(f"fine object {tuple(Xfine.shape)} @0.25um -> GT {gt.shape} (avgpool)  GT axial-std={gt_std:.4f}")

    H2s = {"designed": torch.load(args.designed, map_location=dev)[:T],
           "DMD": baselines.dmd_volume(cfg)[:T]}
    n_vox = 8192; comps = [float(x) for x in args.compressions.split(",")]
    results = {}
    for name, H2raw in H2s.items():
        H2 = matched_dose(H2raw, cfg.dose_mean)                     # one global dose scale
        print(f"[{name}] H2 {tuple(H2.shape)} dose_mean={cfg.dose_mean} col-scale set; building A...")
        t = time.time(); A = build_A_batched(H2, emPSF, args.d_size, args.pattern_batch, args.fwd_chunk, dev)
        A = A.to(dev); print(f"  A {tuple(A.shape)} built {time.time()-t:.0f}s")
        torch.save(A.detach().cpu(), Path(args.save_a) / f"vecA_{name}.pt")
        fwd = ExportForward(H2, emPSF, patch=(4, 4, 4), d_size=args.d_size, chunk=16,
                            dose_mean=None, device=dev)
        Y = add_noise_snr(fwd.forward_fine(Xfine).reshape(-1), args.snr, seed=args.point)  # fine-object Y
        A_j, Y_j = snp.array(A.detach().cpu().numpy()), snp.array(Y.detach().cpu().numpy())
        results[name] = {}
        for c in comps:
            nm = min(A.shape[0], int(round(n_vox / c)))
            for pname, prior in PRESETS.items():
                est, _ = run_pnp(A_j, Y_j, obj_shape=(8, 32, 32), gt=gt_j, n_measure=nm, **prior)
                est = np.asarray(est); m = depth_metrics(gt, est)
                m["shape"] = float(axial_profile(est).std() / (gt_std + 1e-12))
                results[name][(c, pname)] = (est, m)
                print(f"  c={c:g} {pname:5s}: psnr={m['psnr']:.2f}dB centroid_err={m['centroid_err']:.3f} "
                      f"profL1={m['profile_l1']:.4f} shape={m['shape']:.2f}", flush=True)
        del A, A_j; torch.cuda.empty_cache()
        # incremental save: metrics of every coder finished so far (survives a late crash)
        summary = {n: {f"c{c:g}_{p}": {k: results[n][(c, p)][1][k] for k in
                                       ("psnr", "centroid_err", "profile_l1", "shape")}
                       for c in comps for p in PRESETS} for n in results}
        (out / "results.json").write_text(json.dumps(summary, indent=2))
        print(f"  [saved results.json + vecA_{name}.pt]", flush=True)

    # figures: designed vs DMD per (prior, compression)
    for c in comps:
        for pname in ("dncnn", "l1"):
            vols = {"GT": gt}; profs = {}
            for name in H2s:
                est = results[name][(c, pname)][0]; vols[name] = est; profs[name] = axial_profile(est)
            viz.show_mips(vols, title=f"vector {pname} c={c:g}: designed vs DMD (fine-obj Y, avgpool GT)").savefig(
                out / f"mips_{pname}_c{c:g}.png", dpi=140, bbox_inches="tight")
            viz.overlay_axial_profile(axial_profile(gt), profs,
                title=f"vector {pname} c={c:g}: designed vs DMD").savefig(
                out / f"profile_{pname}_c{c:g}.png", dpi=140, bbox_inches="tight")
    print(f"artifacts in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
