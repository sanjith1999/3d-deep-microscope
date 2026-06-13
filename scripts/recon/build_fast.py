#!/usr/bin/env python3
"""
Fast exact operator build, checked against the live forward model, then registered.

    python -u scripts/recon/build_fast.py --coder random --seed 0 --n_layers 1 --name random1L_gap66
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from configs.default import get_config
from pipeline.forward import ExportForward, emission_psf
from operator_assembly import illumination, footprints, assemble


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast exact operator assembly + registration.")
    ap.add_argument("--coder", required=True, choices=["random", "dmd", "designed"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--designed", default=None); ap.add_argument("--name", required=True)
    ap.add_argument("--patterns", type=int, default=256); ap.add_argument("--d_size", type=int, default=8)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--paraxial", action="store_true")
    ap.add_argument("--na_em", type=float, default=None); ap.add_argument("--pattern_idx", default=None, help="json file: scan-pattern subset")
    a = ap.parse_args()
    cfg = get_config(); cfg.device = dev = a.device; cfg.d2nn_paraxial = bool(a.paraxial)
    if a.na_em is not None:
        cfg.na_em = a.na_em
    idx = json.loads(Path(a.pattern_idx).read_text()) if a.pattern_idx else None
    if idx is not None:
        a.patterns = len(idx)
    n_layers = a.n_layers or cfg.n_layers
    meta = {"coder": a.coder, "seed": a.seed, "designed": a.designed, "meas_shape": [a.patterns], "physics": {"n_layers": n_layers, "d2nn_paraxial": bool(a.paraxial)}, "pattern_idx": idx}
    t = time.time(); H2 = illumination(meta, cfg, dev)
    nz, Nc = H2.shape[1], H2.shape[2]
    emPSF = emission_psf(nz, Nc, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
    h = footprints(emPSF, nz, Nc, a.d_size, dev)
    A = assemble(H2, h, "patch", tuple(cfg.recon_patch), dev)
    fwd = ExportForward(H2.to(dev), emPSF, patch=tuple(cfg.recon_patch), d_size=a.d_size, chunk=16, dose_mean=None, device=dev)
    errs = []
    for s in range(3):
        x = torch.rand(nz // cfg.recon_patch[0], Nc // cfg.recon_patch[1], Nc // cfg.recon_patch[2], generator=torch.Generator().manual_seed(s))
        y = fwd.forward(x.to(dev)).reshape(-1).cpu(); errs.append(float((A @ x.reshape(-1) - y).norm() / y.norm()))
    secs = round(time.time() - t)
    print(f"[build_fast] {a.name}: A {tuple(A.shape)} in {secs}s; live-forward rel L2 {errs}", flush=True)
    if max(errs) > 1e-5:
        raise SystemExit("live-forward check failed; not saving")
    dst = _ROOT / "results/recon" / f"vecA_{a.name}.pt"; torch.save(A, dst); del A, fwd, H2; torch.cuda.empty_cache()
    (_ROOT / "results/recon" / f"vecA_{a.name}.fastcheck.json").write_text(json.dumps(
        {"method": "build_fast (illumination x footprint, exact)", "live_forward_rel_l2": errs, "secs": secs}, indent=1))
    cmd = [sys.executable, "-u", str(_ROOT / "scripts/recon/build_operator.py"), "--coder", a.coder, "--seed", str(a.seed),
           "--name", a.name, "--patterns", str(a.patterns), "--d_size", str(a.d_size), "--register_only"]
    if a.n_layers:
        cmd += ["--n_layers", str(a.n_layers)]
    if a.designed:
        cmd += ["--designed", a.designed]
    if a.paraxial:
        cmd += ["--paraxial"]
    if a.na_em is not None:
        cmd += ["--na_em", str(a.na_em)]
    if a.pattern_idx:
        cmd += ["--pattern_idx", a.pattern_idx]
    return subprocess.call(cmd, cwd=_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
