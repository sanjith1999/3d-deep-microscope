#!/usr/bin/env python3
"""
Build one coder's explicit operator A in checkpointed pattern batches and register it.

    python -u scripts/recon/build_operator.py --coder random --seed 0 --name random_gap66
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from configs.default import get_config
from coder import baselines
from pipeline.forward import ExportForward, emission_psf, matched_dose, reduce_object
from pipeline.specimens import NAMES as SPECIMENS, make_specimen


def coder_H2(coder, cfg, designed_path, seed, T, dev, idx=None):
    if idx is not None:                               # an explicit subset of the angle scan
        if coder == "dmd":
            return baselines.dmd_volume(cfg, seed=seed)[idx]
        with torch.no_grad():
            m = baselines.d2nn_model(cfg, seed=seed, coder_path=designed_path if coder == "designed" else None)
            return m(torch.tensor(idx, device=dev)).detach()
    if coder == "designed":
        # regenerate H2 from the trained phases at cfg's pattern count (may differ from design run)
        return baselines.designed_volume(cfg, designed_path)[:T]
    if coder == "dmd":
        return baselines.dmd_volume(cfg, seed=seed)[:T]
    if coder == "random":
        return baselines.random_d2nn_volume(cfg, seed=seed)[:T]
    raise ValueError(coder)


def build_A_batched(H2, emPSF, d_size, pb, patch, dev, parts_dir=None, name="", max_new=None, stop_file=None):
    """Build A in row-batches of `pb` patterns (H2 pre-normalised; dose_mean=None)."""
    T = H2.shape[0]; new = 0
    have = {}
    if parts_dir:
        for f in Path(parts_dir).glob(f"{name}_p*-*.pt"):
            a, b = f.stem[len(name) + 2:].split("-"); have[(int(a), int(b))] = f
    def covered(s, e):
        """Saved parts tiling [s, e) exactly, in order, or None."""
        out, cur = [], s
        while cur < e:
            nxt = [(a, b) for (a, b) in have if a == cur and b <= e]
            if not nxt:
                return None
            ab = max(nxt, key=lambda t: t[1]); out.append(have[ab]); cur = ab[1]
        return out
    parts = []
    for s in range(0, T, pb):
        e = min(s + pb, T)
        files = covered(s, e)
        if files:
            parts += [torch.load(f, map_location="cpu") for f in files]
            print(f"  patterns {s}:{e} -> loaded cached ({len(files)} part(s))", flush=True)
            continue
        if (max_new is not None and new >= max_new) or (stop_file and Path(stop_file).exists()):
            print(f"  paused before patterns {s}:{e} ({new} new batch(es) this run). Re-run the same command to resume.",
                  flush=True)
            return None
        pp = (Path(parts_dir) / f"{name}_p{s:03d}-{e:03d}.pt") if parts_dir else None
        fwd = ExportForward(H2[s:s + pb], emPSF, patch=patch, d_size=d_size,
                            chunk=0, dose_mean=None, device=dev)
        ts = time.time(); part = fwd.build_A().cpu()
        if pp is not None:
            torch.save(part, pp)
        parts.append(part); new += 1
        print(f"  patterns {s}:{e} -> rows {part.shape[0]} ({time.time()-ts:.0f}s)", flush=True)
        del fwd; torch.cuda.empty_cache()
    return torch.cat(parts, 0)


@torch.no_grad()
def register(args, cfg, H2, emPSF, patch, dev, A_path, A_shape, secs=None):
    ops = Path(args.out) / "ops"; entry = ops / args.name; entry.mkdir(parents=True, exist_ok=True)
    fwd = ExportForward(H2, emPSF, patch=patch, d_size=args.d_size, chunk=16, dose_mean=None, device=dev)
    Y, gts = {}, {}
    for sp in SPECIMENS:
        X = make_specimen(sp, cfg).to(dev)
        Y[sp] = fwd.forward_fine(X).reshape(-1).float().cpu()
        gts[sp] = reduce_object(X, patch).float().cpu()
    torch.save(Y, entry / "Y.pt")
    sp = ops / "specimens.pt"                       # merge: keep ground truths other operators registered (e.g. the cohort)
    old = torch.load(sp)["gt"] if sp.exists() else {}
    torch.save({"gt": {**old, **gts}, "patch": patch}, sp)
    del fwd; torch.cuda.empty_cache()
    light = baselines.relay_light(cfg, args.coder, seed=args.seed, coder_path=args.designed)
    Nc = cfg.n_fine // cfg.bin_factor
    meta = {"name": args.name, "coder": args.coder, "seed": args.seed, "designed": args.designed,
            "A": str(A_path).replace("\\", "/"), "A_shape": list(A_shape),
            "patterns": args.patterns, "pattern_idx": getattr(args, "idx", None), "d_size": args.d_size, "patch": list(patch),
            "recon_shape": [cfg.nz // patch[0], Nc // patch[1], Nc // patch[2]],
            "meas_shape": [args.patterns, Nc // args.d_size, Nc // args.d_size],
            "dose_mean": cfg.dose_mean, "light_kept": light, "specimens": list(Y),
            "physics": {k: getattr(cfg, k) for k in ("wavelength_um", "na_ex", "na_em", "neuron_pitch_um",
                        "n_layers", "delta_z_um", "d2nn_out_dist_um", "pitch_fine", "bin_factor", "dz_um", "nz",
                        "d2nn_paraxial")},
            "build_secs": secs, "registered": time.strftime("%Y-%m-%d %H:%M:%S")}
    (entry / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{args.name}] registered -> {entry} (light kept {light:.4f}, specimens {list(Y)})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and save one coder's operator A.")
    ap.add_argument("--coder", required=True, choices=["designed", "dmd", "random"])
    ap.add_argument("--designed", default=None, help="speckle_volume.pt (required for --coder designed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True, help="output label -> results/recon/vecA_<name>.pt")
    ap.add_argument("--patterns", type=int, default=256)
    ap.add_argument("--pattern_batch", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=None,
                    help="override the number of D2NN phase layers (1 = a single phase plate 6.6 um before the "
                         "object plane; the layer ablation). Random coders only; a designed coder follows its checkpoint")
    ap.add_argument("--pattern_idx", default=None,
                    help="json list (file) of scan-pattern indices to build instead of the first --patterns")
    ap.add_argument("--max_batches", type=int, default=None,
                    help="compute at most this many new batches, then exit (resume by re-running)")
    ap.add_argument("--stop_file", default=None,
                    help="pause between batches when this file exists (default: <out>/_parts/<name>.stop)")
    ap.add_argument("--d_size", type=int, default=8, help="detector pooling factor; analysis N/d_size per side")
    ap.add_argument("--out", default="results/recon")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--na_em", type=float, default=None, help="override the detection (emission) NA (default cfg.na_em = 0.1)")
    ap.add_argument("--paraxial", action="store_true",
                    help="paraxial (Fresnel) D2NN propagation: the macro intermediate-image layout (cfg.d2nn_paraxial)")
    ap.add_argument("--register_only", action="store_true",
                    help="skip the build; register an existing vecA_<name>.pt for the notebooks")
    args = ap.parse_args()

    cfg = get_config(); cfg.device = args.device; dev = args.device
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    cfg.d2nn_paraxial = bool(args.paraxial)
    if args.na_em is not None:
        cfg.na_em = args.na_em
    args.idx = json.loads(Path(args.pattern_idx).read_text()) if args.pattern_idx else None
    if args.idx is not None:
        args.patterns = len(args.idx)
    T, Nc, nz = args.patterns, cfg.n_fine // cfg.bin_factor, cfg.nz
    emPSF = emission_psf(nz, Nc, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    H2 = matched_dose(coder_H2(args.coder, cfg, args.designed, args.seed, T, dev, args.idx), cfg.dose_mean)
    torch.cuda.empty_cache()   # release the ~4.5 GB exPSF otf held by the H2-regen model before build_A
    patch = tuple(cfg.recon_patch)
    dst = Path(args.out) / f"vecA_{args.name}.pt"
    if args.register_only:
        if not dst.exists():
            raise SystemExit(f"{dst} not found -- build it first")
        shape = tuple(torch.load(dst, map_location="cpu", mmap=True).shape)
        register(args, cfg, H2, emPSF, patch, dev, dst, shape)
        return 0
    rz, rx, ry = nz // patch[0], Nc // patch[1], Nc // patch[2]
    print(f"[{args.name}] coder={args.coder} H2 {tuple(H2.shape)} patch={patch} d_size={args.d_size} "
          f"-> recon ({rz},{rx},{ry})={rz*rx*ry} voxels, rows {T}x{(Nc//args.d_size)**2}; building A...", flush=True)
    parts_dir = Path(args.out) / "_parts"; parts_dir.mkdir(parents=True, exist_ok=True)
    stop = args.stop_file or str(parts_dir / f"{args.name}.stop")
    t = time.time(); A = build_A_batched(H2, emPSF, args.d_size, args.pattern_batch, patch, dev,
                                         parts_dir=parts_dir, name=args.name, max_new=args.max_batches,
                                         stop_file=stop)
    if A is None:
        return 0
    torch.save(A, dst); secs = round(time.time() - t)
    print(f"[{args.name}] A {tuple(A.shape)} built {secs}s -> {dst}", flush=True)
    shape = tuple(A.shape); del A
    register(args, cfg, H2, emPSF, patch, dev, dst, shape, secs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
