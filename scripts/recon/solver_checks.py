#!/usr/bin/env python3
"""
Solver checks: slice orientation of each denoiser wrapper and run-to-run reproducibility.
"""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import jax, scico
import scico.numpy as snp
from scico import functional

from pipeline.reconstruct import RECON, PRIOR_PRESETS, DnCNN_YZ, DnCNN_ZX


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def test_volume(shape=(16, 48, 48), seed=0):
    """Smooth blobs at asymmetric positions plus mild noise, in [0, 1]; no axis symmetry."""
    rng = np.random.default_rng(seed)
    z, x, y = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    v = np.zeros(shape, np.float32)
    for (cz, cx, cy, s) in [(3, 10, 30, 3.0), (11, 35, 12, 4.0), (7, 20, 40, 2.5), (13, 40, 38, 3.5)]:
        v += np.exp(-((z - cz) ** 2 + (x - cx) ** 2 / 2 + (y - cy) ** 2 / 3) / (2 * s ** 2))
    v += 0.05 * x / shape[1] + 0.03 * (y / shape[2]) ** 2            # asymmetric ramps
    v = v / v.max()
    return np.clip(v + 0.05 * rng.standard_normal(shape).astype(np.float32), 0, 1)


def main() -> int:
    variant = RECON["variant"]
    v = test_volume()
    base = functional.DnCNN(variant=variant)
    den2 = lambda im: np.asarray(base.prox(snp.array(im), 1.0))       # one 2D image
    explicit = {
        "xy": np.stack([den2(v[k]) for k in range(v.shape[0])], 0),
        "yz": np.stack([den2(v[:, i, :]) for i in range(v.shape[1])], 1),
        "zx": np.stack([den2(v[:, :, j]) for j in range(v.shape[2])], 2),
    }
    wrappers = {"DnCNN (default, used as 'xy')": functional.DnCNN(variant=variant),
                "DnCNN_YZ (named 'yz')": DnCNN_YZ(variant=variant),
                "DnCNN_ZX (named 'zx')": DnCNN_ZX(variant=variant)}
    orient, mapping = {}, {}
    for name, w in wrappers.items():
        out = np.asarray(w.prox(snp.array(v), 1.0))
        orient[name] = {o: rel(out, e) for o, e in explicit.items()}
        mapping[name] = min(orient[name], key=orient[name].get)
        print(f"[orientation] {name:30s} " + "  ".join(f"{o}: {d:.2e}" for o, d in orient[name].items())
              + f"   -> denoises physical {mapping[name]} slices", flush=True)

    outs = {lam: np.asarray(base.prox(snp.array(v[0]), lam)) for lam in (0.2, 1.0, 5.0)}
    scale = {f"lam{a}_vs_lam1": rel(outs[a], outs[1.0]) for a in (0.2, 5.0)}
    ignores = all(d < 1e-6 for d in scale.values())
    print(f"[scale] {variant}: " + "  ".join(f"{k} rel diff {d:.2e}" for k, d in scale.items())
          + f"   -> {'IGNORES' if ignores else 'USES'} the proximal scale", flush=True)

    weights = []
    for f in sorted(Path(scico.__file__).parent.rglob("*")):
        if f.is_file() and "dncnn" in str(f).lower() and f.suffix not in (".py", ".pyc"):
            weights.append({"file": str(f.relative_to(Path(scico.__file__).parent)),
                            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    rep = {"orientation_rel_l2": orient, "orientation_map": mapping,
           "test_volume": "16x48x48 (z,x,y), 4 Gaussian blobs + ramps + 0.05 noise, seed 0, clipped to [0,1]",
           "proximal_scale": {"variant": variant, "rel_diff": scale, "ignores_scale": ignores},
           "RECON": RECON, "PRIOR_PRESETS": PRIOR_PRESETS,
           "versions": {"python": platform.python_version(), "jax": jax.__version__, "scico": scico.__version__,
                        "numpy": np.__version__, "devices": [str(d) for d in jax.devices()]},
           "dncnn_weight_files": weights}
    out = _ROOT / "results/recon/solver_checks.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(f"[solver_checks] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
