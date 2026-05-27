#!/usr/bin/env python3
"""
Does the surrogate score predict reconstruction quality? Scores a family of coders both ways.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scico.numpy as snp
from scipy.stats import spearmanr, pearsonr

from configs.default import get_config
from coder import baselines
from pipeline.forward import ExportForward, emission_psf, bead_phantom, reduce_object, add_noise_snr
from pipeline.reconstruct import run_pnp

BEADS = [(1, 10, 10), (2, 22, 10), (4, 16, 16), (5, 10, 22), (6, 22, 22)]


def main() -> int:
    ap = argparse.ArgumentParser(description="E3 proxy-fidelity: surrogate score vs PSNR3D.")
    ap.add_argument("--run_dir", required=True, help="E2 run dir (coder_final.pt + ckpt_*.pt + speckle_volume.pt)")
    ap.add_argument("--patterns", type=int, default=32)
    ap.add_argument("--d_size", type=int, default=4)
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--object", default=None, help="reduced object Xr (rz,rx,ry); default = bead phantom")
    ap.add_argument("--out", default="results/coder/e3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = get_config(); cfg.device = args.device
    dev, T = args.device, args.patterns
    nz, N = cfg.nz, cfg.n_fine // cfg.bin_factor
    focus = nz // 2 - 1
    rd = Path(args.run_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / stamp; out.mkdir(parents=True, exist_ok=True)

    # --- assemble the coder family (name -> H2 producer) ---
    def load_ckpt(p): return baselines.designed_volume(cfg, p)
    family = [("DMD", lambda: baselines.dmd_volume(cfg, seed=0))]
    for s in (0, 1, 2):
        family.append((f"random-s{s}", (lambda s=s: baselines.random_d2nn_volume(cfg, seed=s))))
    for ck in ("ckpt_00500.pt", "ckpt_01000.pt", "ckpt_01500.pt"):
        if (rd / ck).exists():
            step = int(ck.split("_")[1].split(".")[0])
            family.append((f"partial-{step}", (lambda p=rd / ck: load_ckpt(p))))
    family.append(("designed", lambda: torch.load(rd / "speckle_volume.pt", map_location=dev)))

    emPSF = emission_psf(nz, N, 0.25, 0.25, cfg.na_em, device=dev)
    if args.object:
        Xr_raw = torch.load(args.object, map_location=dev).float()
        if Xr_raw.ndim == 2:
            Xr_raw = Xr_raw[0]
        Xr = Xr_raw.reshape(nz // 4, N // 4, N // 4)
        Xr = Xr / (Xr.max() + 1e-12)
    else:
        Xr = reduce_object(bead_phantom(nz, N, BEADS, radius_um=1.0, device=dev))
    gt_j = snp.array(Xr.detach().cpu().numpy())
    print(f"E3 | family={len(family)} patterns={T} d_size={args.d_size} snr={args.snr} "
          f"object={args.object or 'beads'}")

    rows = []
    for name, make in family:
        try:
            H2 = make()[:T]
            score = baselines.score_volume(H2, focus, cfg.dz_um)
            fwd = ExportForward(H2, emPSF, patch=(4, 4, 4), d_size=args.d_size, device=dev)
            t = time.time(); A = fwd.build_A(); bt = time.time() - t
            Y = add_noise_snr(A @ Xr.reshape(-1), args.snr, seed=0)
            _, psnr = run_pnp(snp.array(A.detach().cpu().numpy()),
                              snp.array(Y.detach().cpu().numpy()), obj_shape=tuple(Xr.shape), gt=gt_j)
            rows.append({"name": name, "sym": score["sym"], "coh_res": score["coherence_res"],
                         "offdiag": score["offdiag"], "psnr3d": psnr, "A_build_s": bt})
            print(f"{name:14s}: sym={score['sym']:.3f} coh_res={score['coherence_res']:.3f} "
                  f"PSNR3D={psnr:.2f}dB ({bt:.0f}s)")
            del A, fwd; torch.cuda.empty_cache()
        except Exception as e:  # keep the family run alive; log the failure
            print(f"{name:14s}: FAILED {type(e).__name__}: {e}")
            rows.append({"name": name, "error": f"{type(e).__name__}: {e}"})
        (out / "summary.json").write_text(json.dumps({"config": {"patterns": T, "d_size": args.d_size,
                                          "snr_db": args.snr, "beads": BEADS}, "rows": rows}, indent=2))

    # --- correlation plot ---
    ok = [r for r in rows if "psnr3d" in r]
    if len(ok) >= 3:
        sym = np.array([r["sym"] for r in ok]); coh = np.array([r["coh_res"] for r in ok])
        psnr = np.array([r["psnr3d"] for r in ok]); names = [r["name"] for r in ok]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for a, x, lbl in ((ax[0], sym, "sym(+d/-d)"), (ax[1], coh, "coherence beyond ~1um")):
            a.scatter(x, psnr, c="C0")
            for xi, yi, n in zip(x, psnr, names):
                a.annotate(n, (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
            sp = spearmanr(x, psnr).correlation; pe = pearsonr(x, psnr)[0]
            a.set_xlabel(lbl); a.set_ylabel("PSNR3D (dB)")
            a.set_title(f"{lbl} vs PSNR  (Spearman {sp:+.2f}, Pearson {pe:+.2f})")
            a.grid(alpha=0.3)
        fig.suptitle("E3 proxy fidelity: surrogate score predicts reconstruction quality")
        fig.tight_layout(); fig.savefig(out / "proxy_fidelity.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Spearman(sym,PSNR)={spearmanr(sym, psnr).correlation:+.2f}  "
              f"Spearman(coh_res,PSNR)={spearmanr(coh, psnr).correlation:+.2f}")
    print(f"artifacts in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
