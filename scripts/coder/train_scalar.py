#!/usr/bin/env python3
"""
Design the two-layer scalar coder against the axial decorrelation objective.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse, json, time, sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from configs.default import get_config
from coder.scalar import build_scalar_coder
from coder.losses import decorrelation_loss, smoothness_loss, coherence_score
from coder.utils import seed_everything


def main() -> int:
    ap = argparse.ArgumentParser(description="Surrogate-design the scalar 2-layer coder.")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--train_patterns", type=int, default=32, help="angle-scan subset for design")
    ap.add_argument("--spread_chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument("--out", default="results/coder/scalar")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = get_config(); cfg.device = args.device; seed_everything(args.seed)
    dev = args.device
    stamp = time.strftime("%Y%m%d_%H%M%S"); out = Path(args.out) / stamp; out.mkdir(parents=True, exist_ok=True)

    coder = build_scalar_coder(cfg)
    opt = torch.optim.Adam(coder.parameters(), lr=args.lr)
    focus = coder.focus_idx
    w = cfg.loss_weights
    tp = min(args.train_patterns, coder.n_patterns)
    print(f"design scalar coder: patterns(train)={tp}/{coder.n_patterns} focus={focus} "
          f"weights decorr={w['decorr']} sym={w['sym']} smooth={w['smooth']}  steps={args.steps}")

    hist = []; t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        field = coder.d2nn(coder.inputs[:tp])
        H2 = coder.exc.spread(field, chunk=args.spread_chunk, ckpt=True)   # (tp, nz, N, N)
        off, sym = decorrelation_loss(H2, focus)
        smooth = smoothness_loss(torch.stack(coder.phases))
        total = w["decorr"] * off + w["sym"] * sym + w["smooth"] * smooth
        total.backward(); opt.step()
        if step % args.print_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                coh = coherence_score(H2)
            hist.append({"step": step, "total": float(total), "offdiag": float(off),
                         "sym": float(sym), "smooth": float(smooth), "coherence": coh})
            print(f"[{step:4d}/{args.steps}] total={float(total):.4e} off={float(off):.3e} "
                  f"sym={float(sym):.3e} smooth={float(smooth):.3e} coh={coh:.3f}")

    wall = time.time() - t0
    with torch.no_grad():
        H2_full = coder(chunk=32).detach().cpu()          # all 128 patterns for the operator build
    torch.save({"phases": [p.detach().cpu() for p in coder.phases], "config": asdict(cfg)},
               out / "coder_final.pt")
    torch.save(H2_full, out / "speckle_volume.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "manifest.json").write_text(json.dumps(
        {"wall_seconds": wall, "train_patterns": tp, "focus": focus, "steps": args.steps,
         "final": hist[-1] if hist else None}, indent=2))
    print(f"designed coder -> {out}  ({wall:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
