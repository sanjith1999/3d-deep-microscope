#!/usr/bin/env python3
"""
Physical-grid reconstruction test, stage 2: PnP reconstruction and depth metrics.
"""
from __future__ import annotations
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # share GPU with the 5.4 GB A
import argparse, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scico.numpy as snp

from pipeline.forward import add_noise_snr
from pipeline.reconstruct import run_pnp, PRIOR_PRESETS
from pipeline.metrics import axial_profile, score, depth_blind_oracle, bead_report

SHAPE = (16, 48, 48)


def figure(name, gt, recs, out):
    items = [("GT", gt)] + list(recs.items())
    fig, ax = plt.subplots(3, len(items), figsize=(3.2 * len(items), 9))
    for j, (t, v) in enumerate(items):
        ax[0, j].imshow(v.max(0), cmap="magma"); ax[0, j].set_title(f"{t}\nxy MIP")
        ax[1, j].imshow(v.max(1), cmap="magma", aspect=48 / 16 / 3); ax[1, j].set_title("xz MIP (z down)")
        ax[2, j].plot(axial_profile(gt), "k--", label="GT"); ax[2, j].plot(axial_profile(v), label=t)
        ax[2, j].set_xlabel("depth plane"); ax[2, j].legend(fontsize=7)
        for a in ax[:2, j]: a.axis("off")
    fig.suptitle(f"{name}: designed physical D2NN (16x48x48 @ 0.64 um)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruct with the physical designed operator.")
    ap.add_argument("--A", default="results/recon/vecA_designed_phys.pt")
    ap.add_argument("--inputs", default="results/recon/phys_recon_inputs.pt")
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/recon/phys_recon")
    args = ap.parse_args()
    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)

    t = time.time()
    A_np = torch.load(args.A, map_location="cpu", mmap=True).numpy()
    A = snp.array(A_np); del A_np
    print(f"A {A.shape} on device in {time.time()-t:.0f}s", flush=True)
    inp = torch.load(args.inputs, map_location="cpu")
    beads = [tuple(b) for b in inp["beads_list"]]

    report = {}
    for name in ("beads", "neural_cell"):
        gt = inp[name]["gt"].numpy(); Yc = inp[name]["Y"]
        mism = float(np.linalg.norm(Yc.numpy() - np.asarray(A @ snp.array(gt.reshape(-1))))
                     / (np.linalg.norm(Yc.numpy()) + 1e-12))
        Y = snp.array(add_noise_snr(Yc, args.snr, seed=args.seed).numpy())
        x0 = np.asarray(A.T @ Y).reshape(SHAPE); x0 = np.clip(x0 / (x0.max() + 1e-12), 0, 1)
        recs, res = {}, {"model_mismatch": mism, "gt_axial_std": float(axial_profile(gt).std())}
        res["backprojection"] = score(gt, x0)
        res["depth_blind_oracle"] = score(gt, np.clip(depth_blind_oracle(gt), 0, 1))
        for prior in ("dncnn", "l1"):
            ts = time.time()
            est, _ = run_pnp(A, Y, obj_shape=SHAPE, gt=snp.array(gt), **PRIOR_PRESETS[prior])
            est = np.asarray(est); recs[f"designed-{prior}"] = est
            res[f"designed-{prior}"] = score(gt, est)
            if name == "beads":
                res[f"designed-{prior}"]["beads"] = bead_report(est, beads)
            print(f"[{name}] {prior} done {time.time()-ts:.0f}s", flush=True)
        recs["depth-blind oracle"] = depth_blind_oracle(gt)
        np.savez_compressed(out / f"{name}_recon.npz", gt=gt, **{k.replace(' ', '_'): v for k, v in recs.items()})
        figure(name, gt, recs, out / f"{name}.png")
        report[name] = res

        print(f"\n=== {name} (SNR {args.snr:g} dB)  model-mismatch ||Y-A.GT||/||Y|| = {mism:.3f} ===", flush=True)
        print(f"{'method':22s} {'PSNR':>7s} {'globCentErr':>11s} {'localDepthErr':>13s} {'shape':>6s}", flush=True)
        for k in ("backprojection", "depth_blind_oracle", "designed-dncnn", "designed-l1"):
            m = res[k]
            print(f"{k:22s} {m['psnr']:7.2f} {m['centroid_err']:11.3f} {m['local_depth_err']:13.3f} {m['shape']:6.2f}", flush=True)
        if name == "beads":
            for prior in ("dncnn", "l1"):
                print(f"  beads [{prior}]: " + "  ".join(
                    f"z{b['true_z']}->peak{b['peak_z']} ghost{b['ghost_ratio']:.2f}"
                    for b in res[f"designed-{prior}"]["beads"]), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nartifacts in: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
