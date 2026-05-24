#!/usr/bin/env python3
"""
Design the physical-neuron diffractive coder with the object-plane loss and save its phases and volume.
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from configs.default import get_config
from coder.train import train


def main() -> int:
    ap = argparse.ArgumentParser(description="Design the physical-neuron coder.")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n_thetas", type=int, default=None, help="override angle scan (patterns = n_alphas * n_thetas)")
    ap.add_argument("--pattern_subset", type=int, default=None, help="patterns per step (0 = all)")
    ap.add_argument("--out_dist", type=float, default=None, help="override D2NN output gap (um)")
    ap.add_argument("--loss_plane", choices=["relay", "object"], default=None,
                    help="evaluate the loss after the NA relay cut (relay) or before it (object)")
    ap.add_argument("--loss_dist", type=float, nargs="+", default=None,
                    help="loss-plane distance after the last layer (um): one value = fixed, two = random range")
    ap.add_argument("--n_layers", type=int, default=None, help="override number of learnable D2NN layers")
    ap.add_argument("--loss_mode", choices=["stats", "output"], default=None,
                    help="stats = pattern-statistics loss (v2-v5); output = NA-0.8-output characteristics (v6)")
    ap.add_argument("--init", default=None, help="coder_final.pt to initialise the phases from")
    ap.add_argument("--w", nargs="*", default=[], metavar="KEY=VAL",
                    help="loss-weight overrides, e.g. --w pdf=0 contrast=1")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--out", default="results/coder/physical")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(); cfg.device = args.device; cfg.seed = args.seed
    if args.n_thetas is not None:
        cfg.n_thetas = args.n_thetas
    if args.pattern_subset is not None:
        cfg.training["pattern_subset"] = args.pattern_subset
    if args.out_dist is not None:
        cfg.d2nn_out_dist_um = args.out_dist
    if args.loss_plane is not None:
        cfg.training["loss_plane"] = args.loss_plane
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.loss_mode is not None:
        cfg.training["loss_mode"] = args.loss_mode
    if args.init is not None:
        cfg.training["init_ckpt"] = args.init
    for kv in args.w:
        k, v = kv.split("=", 1)
        if k not in cfg.loss_weights:
            raise SystemExit(f"unknown loss weight {k!r}; known: {sorted(cfg.loss_weights)}")
        cfg.loss_weights[k] = float(v)
    if args.loss_dist is not None:
        cfg.training["loss_dist_um"] = args.loss_dist[0] if len(args.loss_dist) == 1 else list(args.loss_dist[:2])
    cfg.training["num_steps"] = args.steps
    cfg.training["print_every"] = args.print_every
    cfg.training["checkpoint_every"] = 0
    if args.lr is not None:
        cfg.training["lr"] = args.lr

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / stamp
    T = cfg.n_alphas * cfg.n_thetas
    print("=" * 70, flush=True)
    print(f"design | neurons {cfg.neuron_pitch_um}um layers={cfg.n_layers} gap={cfg.d2nn_out_dist_um}um "
          f"fine={cfg.n_fine}@{cfg.pitch_fine} patterns={T} subset={cfg.training['pattern_subset']} "
          f"loss_plane={cfg.training.get('loss_plane', 'relay')} loss_dist={cfg.training.get('loss_dist_um')} "
          f"target={cfg.target['name']} weights={cfg.loss_weights} steps={args.steps}", flush=True)
    print(f"out -> {run_dir}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    result = train(cfg, run_dir)
    m = result["manifest"]
    print("-" * 70, flush=True)
    print(f"DONE {time.time()-t0:.0f}s ({(time.time()-t0)/args.steps:.1f}s/step)  "
          f"peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    if m.get("loss_mode") == "output":
        print(f"final light={m['final_light']:.4f} (min {m['final_light_min']:.4f}) c064={m['final_c064']:.3f} "
              f"axial_corr={m['final_axial_corr']:.3f} diag depth_sym={m['final_diag_depth_sym']:.4f}", flush=True)
    else:
        print(f"final corr={m['final_pattern_corr']:.4f} near={m['final_pattern_near']:.4f} "
              f"nn={m['final_pattern_nn']:.4f} contrast={m['final_contrast']:.3f} (target {m['target_contrast']:.3f}) "
              f"pdf={m['final_pdf']:.4e} diag depth_sym={m['final_diag_depth_sym']:.4f}", flush=True)
    print(f"phases -> {run_dir}/coder_final.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
