#!/usr/bin/env python3
"""
Designed vs DMD across specimens with the saved operators.
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Designed vs DMD across specimens (reuse operators).")
    ap.add_argument("--a_designed", default="results/recon/vecA_designed.pt")
    ap.add_argument("--a_dmd", default="results/recon/vecA_DMD.pt")
    ap.add_argument("--a_random", default=None, help="optional random-D2NN operator to include")
    ap.add_argument("--seed", type=int, default=0, help="random-D2NN seed (matches its build)")
    ap.add_argument("--designed", required=True, help="designed coder speckle_volume.pt (for H2)")
    ap.add_argument("--xdir", default="backup_fyp/D2NN", help="dir with X_<specimen>.pt fine objects")
    ap.add_argument("--specimens", default="neural_cell,blood_cell,mnist")
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--patterns", type=int, default=128)
    ap.add_argument("--d_size", type=int, default=16)
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--compressions", default="1,8")
    ap.add_argument("--out", default="results/recon/repro_specimens")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(); cfg.device = args.device; dev = args.device
    T, Nc, nz = args.patterns, cfg.n_fine // cfg.bin_factor, cfg.nz
    emPSF = emission_psf(nz, Nc, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
    comps = [float(x) for x in args.compressions.split(",")]
    specimens = [s.strip() for s in args.specimens.split(",") if s.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S"); out = Path(args.out) / stamp; out.mkdir(parents=True, exist_ok=True)
    n_vox = 8192

    # operators (saved) + H2 (for the measurement) per coder; dose-normalised once
    H2d = matched_dose(torch.load(args.designed, map_location=dev)[:T], cfg.dose_mean)
    H2m = matched_dose(baselines.dmd_volume(cfg)[:T], cfg.dose_mean)
    coders = {
        "designed": (snp.array(torch.load(args.a_designed, map_location="cpu").float().numpy()),
                     ExportForward(H2d, emPSF, patch=(4, 4, 4), d_size=args.d_size, chunk=16,
                                   dose_mean=None, device=dev)),
        "DMD":      (snp.array(torch.load(args.a_dmd, map_location="cpu").float().numpy()),
                     ExportForward(H2m, emPSF, patch=(4, 4, 4), d_size=args.d_size, chunk=16,
                                   dose_mean=None, device=dev)),
    }
    if args.a_random:
        H2r = matched_dose(baselines.random_d2nn_volume(cfg, seed=args.seed)[:T], cfg.dose_mean)
        coders["random-D2NN"] = (snp.array(torch.load(args.a_random, map_location="cpu").float().numpy()),
                                 ExportForward(H2r, emPSF, patch=(4, 4, 4), d_size=args.d_size,
                                               chunk=16, dose_mean=None, device=dev))
    print(f"loaded operators + H2. coders={list(coders)} specimens={specimens} compressions={comps}", flush=True)

    summary = {}
    for sp in specimens:
        raw = torch.load(f"{args.xdir}/X_{sp}.pt", map_location="cpu")
        Xfine = raw[args.point].reshape(nz, Nc, Nc).float().to(dev); del raw
        gt = reduce_object(Xfine).detach().cpu().numpy()
        gt_j = snp.array(gt); gt_std = float(axial_profile(gt).std())
        print(f"\n=== {sp}: GT {gt.shape} axial-std={gt_std:.4f} ===", flush=True)
        summary[sp] = {"gt_axial_std": gt_std}
        recon_vols = {c: {"GT": gt} for c in comps}
        for name, (A_j, fwd) in coders.items():
            Y = add_noise_snr(fwd.forward_fine(Xfine).reshape(-1), args.snr, seed=args.point)
            Y_j = snp.array(Y.detach().cpu().numpy())
            for c in comps:
                nm = min(A_j.shape[0], int(round(n_vox / c)))
                for pname, prior in PRESETS.items():
                    est, _ = run_pnp(A_j, Y_j, obj_shape=(8, 32, 32), gt=gt_j, n_measure=nm, **prior)
                    est = np.asarray(est); m = depth_metrics(gt, est)
                    m["shape"] = float(axial_profile(est).std() / (gt_std + 1e-12))
                    summary[sp][f"{name}_c{c:g}_{pname}"] = {k: m[k] for k in
                        ("psnr", "centroid_err", "profile_l1", "shape")}
                    if pname == "dncnn":
                        recon_vols[c][name] = est
                    print(f"  {name:9s} c={c:g} {pname:5s}: psnr={m['psnr']:.2f} "
                          f"cErr={m['centroid_err']:.3f} shape={m['shape']:.2f}", flush=True)
        for c in comps:
            profs = {n: axial_profile(v) for n, v in recon_vols[c].items() if n != "GT"}
            viz.overlay_axial_profile(axial_profile(gt), profs,
                title=f"{sp} c={c:g} (DnCNN): designed vs DMD").savefig(
                out / f"profile_{sp}_c{c:g}.png", dpi=140, bbox_inches="tight")
            viz.show_mips(recon_vols[c], title=f"{sp} c={c:g} (DnCNN)").savefig(
                out / f"mips_{sp}_c{c:g}.png", dpi=140, bbox_inches="tight")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nartifacts in: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
