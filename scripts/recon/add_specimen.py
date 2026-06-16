#!/usr/bin/env python3
"""
Measure additional specimens with already registered operators, without rebuilding them.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from configs.default import get_config
from pipeline.forward import ExportForward, emission_psf, matched_dose, reduce_object
from pipeline.specimens import make_specimen, COHORT
from build_operator import coder_H2


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure a new specimen through registered operators.")
    ap.add_argument("--ops", required=True)
    ap.add_argument("--specimen", required=True, help="name, comma list, or 'cohort' (pipeline.specimens.COHORT)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ops = _ROOT / "results/recon/ops"
    for op in a.ops.split(","):
        meta = json.loads((ops / op / "meta.json").read_text())
        if meta.get("iid_plates"):
            raise SystemExit(f"{op}: per-frame plates cannot be regenerated here; measure specimens in build_iid.py")
        cfg = get_config(); cfg.device = dev = a.device
        cfg.n_layers = meta.get("physics", {}).get("n_layers", cfg.n_layers)
        cfg.na_em = meta.get("physics", {}).get("na_em", cfg.na_em)                    # detection NA of THIS operator
        cfg.d2nn_paraxial = meta.get("physics", {}).get("d2nn_paraxial", False)
        T = meta["meas_shape"][0]; idx = meta.get("pattern_idx")
        t = time.time()
        H2 = matched_dose(coder_H2(meta["coder"], cfg, meta.get("designed"), meta.get("seed", 0), T, dev, idx),
                          cfg.dose_mean)
        torch.cuda.empty_cache()
        Nc, nz, patch = cfg.n_fine // cfg.bin_factor, cfg.nz, tuple(meta["patch"])
        emPSF = emission_psf(nz, Nc, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
        fwd = ExportForward(H2, emPSF, patch=patch, d_size=meta["d_size"], chunk=16, dose_mean=None, device=dev)
        names = COHORT if a.specimen == "cohort" else a.specimen.split(",")
        Y = torch.load(ops / op / "Y.pt"); sp = torch.load(ops / "specimens.pt")
        for name in names:
            X = make_specimen(name, cfg, root=_ROOT).to(dev)
            y = fwd.forward_fine(X).reshape(-1).float().cpu()
            Y[name] = y; sp["gt"][name] = reduce_object(X, patch).float().cpu()
            print(f"[add_specimen] {op}: {name} measured ({y.numel()} values, mean {float(y.mean()):.4g}) "
                  f"at {time.time() - t:.0f}s", flush=True)
        torch.save(Y, ops / op / "Y.pt"); torch.save(sp, ops / "specimens.pt")
        meta["specimens"] = sorted(set(meta.get("specimens", [])) | set(names))
        (ops / op / "meta.json").write_text(json.dumps(meta, indent=2))
        del H2, fwd; torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
