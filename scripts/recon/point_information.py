#!/usr/bin/env python3
"""
Point-emitter information on the analysis grid: axial Cramer-Rao bound and mirror discriminability d'
for the DMD and the one- and two-layer plates, at full scan and k-pattern subsets.
"""
from __future__ import annotations
import argparse, copy, json, math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from configs.default import get_config
from coder import baselines
from coder.psf import conv_3D
from pipeline.forward import emission_psf, matched_dose

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def illumination(coder, cfg, seed=0):
    """Dose-normalised H2 (T, nz, N, N) exactly as build_operator makes it (matched over all T patterns)."""
    if coder == "dmd":
        H2 = baselines.dmd_volume(cfg, seed=seed)
    else:
        c = copy.copy(cfg); c.n_layers = 1 if coder.startswith("random1") else cfg.n_layers
        c.d2nn_paraxial = coder.endswith("p")            # random1p / random2p: macro (paraxial) implementation
        with torch.no_grad():
            H2 = baselines.d2nn_model(c, seed=seed)().detach()
    return matched_dose(H2, cfg.dose_mean).float()


def footprints(emPSF, zs, lat, N, d, w=2, batch=32):
    nz = emPSF.shape[1]; m = nz // 2; dev = emPSF.device
    pos = [(z, x, y) for z in zs for (x, y) in lat]
    out = []
    for s in range(0, len(pos), batch):
        X = torch.zeros(len(pos[s:s + batch]), nz, N, N, device=dev)
        for i, (z, x, y) in enumerate(pos[s:s + batch]):
            X[i, z, x, y] = 1.0
        img = conv_3D(emPSF, X.to(torch.complex64), w).real[:, m]
        out.append(F.avg_pool2d(img.unsqueeze(1), d).flatten(1).cpu())
    return torch.cat(out, 0).double().numpy().reshape(len(zs), len(lat), -1)        # (nzs, nl, Nd*Nd)


def fisher(a, da):
    """Brightness-marginalised Fisher information per (b/sigma)^2; a, da: (..., M) -> (...)."""
    aa = (a * a).sum(-1); dd = (da * da).sum(-1); ad = (a * da).sum(-1)
    return np.maximum(dd - ad ** 2 / np.maximum(aa, 1e-300), 0.0)


def summarise(v):
    v = v[np.isfinite(v)]
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))} if v.size else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Point-emitter CRB and mirror discriminability per coder.")
    ap.add_argument("--coders", default="random2,random1,dmd")
    ap.add_argument("--ks", default="8,16,32,64,128,256")
    ap.add_argument("--rules", default="first,random,spread")
    ap.add_argument("--subsets", default="results/recon/compression/subsets.json")
    ap.add_argument("--subsets_op", default="random_gap66", help="whose subsets.json entry gives the pattern subsets")
    ap.add_argument("--q", type=float, default=25.0)
    ap.add_argument("--ref_op", default="random_gap66", help="registered operator whose Y fixes sigma")
    ap.add_argument("--ref", default="blood_cell")
    ap.add_argument("--lat_step", type=int, default=8, help="analysis-grid lateral step (8 -> 12 x 12 emitters)")
    ap.add_argument("--d_size", type=int, default=8)
    ap.add_argument("--out", default="results/paper/information")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the element (DMD pattern seed)")
    ap.add_argument("--tag", default="", help="suffix for the output json (e.g. _seed1)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = get_config(); cfg.device = dev = a.device
    out = _ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    N, nz = cfg.n_fine // cfg.bin_factor, cfg.nz
    pitch, dz = cfg.pitch_scalar, cfg.dz_um
    fz = 15                                                      # excitation focus (VectorialExcitation.focus_idx)
    zs = list(range(nz))
    xs = list(range(a.lat_step // 2, N, a.lat_step)); lat = [(x, y) for x in xs for y in xs]
    latp = [(min(x + 1, N - 1), y) for (x, y) in lat]; latm = [(max(x - 1, 0), y) for (x, y) in lat]

    emPSF = emission_psf(nz, N, dz, pitch, cfg.na_em, device=dev)
    t = time.time()
    h = footprints(emPSF, zs, lat, N, a.d_size); hp = footprints(emPSF, zs, latp, N, a.d_size)
    hm = footprints(emPSF, zs, latm, N, a.d_size)
    sharp = [(float((h[z] ** 2).sum(-1).mean() / (h[z].sum(-1) ** 2).mean())) for z in zs]   # concentration
    print(f"[point] footprints {h.shape} in {time.time() - t:.0f}s; sharpest detection plane {int(np.argmax(sharp))} "
          f"(excitation focus {fz})", flush=True)

    ops = _ROOT / "results/recon/ops"
    sigma0 = float(torch.load(ops / a.ref_op / "Y.pt")[a.ref].numpy().mean()) * 10 ** (-a.q / 10)
    subs = json.loads((_ROOT / a.subsets).read_text()).get(a.subsets_op, {}) if (_ROOT / a.subsets).exists() else {}
    report = {"definition": __doc__.split("    python")[0].strip(), "element_seed": a.seed, "q": a.q, "sigma_frame": sigma0, "focus_plane": fz,
              "dz_um": dz, "pitch_um": pitch, "sharpness_by_plane": sharp, "results": [], "symmetry": {}}
    for coder in a.coders.split(","):
        t = time.time()
        H2 = illumination(coder, cfg, seed=a.seed)
        T = H2.shape[0]
        sym = [float((H2[:, fz + d] - H2[:, fz - d]).norm() / H2[:, fz + d].norm()) for d in range(1, 8)]
        report["symmetry"][coder] = sym
        def ill(lt):
            xt = torch.tensor([x for x, _ in lt]); yt = torch.tensor([y for _, y in lt])
            return H2[:, :, xt, yt].double().cpu().numpy()                       # (T, nz, nl)
        I0, Ip, Im = ill(lat), ill(latp), ill(latm)
        del H2; torch.cuda.empty_cache()
        print(f"[point] {coder}: illumination T={T} in {time.time() - t:.0f}s; asymmetry about focus "
              f"(rel. diff of planes 15+-d, d=1..7): " + " ".join(f"{v:.3f}" for v in sym), flush=True)
        sets = []
        for k in [int(x) for x in a.ks.split(",")]:
            if k >= T:
                sets.append(("all", 0, k, list(range(T))))
                continue
            for rule in a.rules.split(","):
                if coder == "dmd" and rule not in ("first", "random"):
                    continue
                for draw, pats in enumerate(subs.get(str(k), {}).get(rule, [])):
                    sets.append((rule, draw, k, pats))
        for (rule, draw, k, pats) in sets:
            P = np.array(sorted(pats))
            J0, Jp, Jm = I0[P], Ip[P], Im[P]                       # (k, nz, nl)
            dot = lambda I1, h1, I2, h2: (I1 * I2).sum(0) * (h1 * h2).sum(-1)       # -> (nl,)
            for dose in ("frame", "total"):
                sig = sigma0 * (math.sqrt(k / 256) if dose == "total" else 1.0)
                cz = np.full((nz, len(lat)), np.nan); dm = np.full_like(cz, np.nan); cx = np.full_like(cz, np.nan)
                for z in range(nz):
                    u = (J0[:, z], h[z]); aa = dot(*u, *u)
                    if 0 < z < nz - 1:
                        up, um = (J0[:, z + 1], h[z + 1]), (J0[:, z - 1], h[z - 1])
                        dd = (dot(*up, *up) + dot(*um, *um) - 2 * dot(*up, *um)) / (2 * dz) ** 2
                        ad = (dot(*u, *up) - dot(*u, *um)) / (2 * dz)
                        cz[z] = 1 / np.sqrt(np.maximum((dd - ad ** 2 / np.maximum(aa, 1e-300)) / sig ** 2, 1e-300))
                    xp, xm = (Jp[:, z], hp[z]), (Jm[:, z], hm[z])
                    dd = (dot(*xp, *xp) + dot(*xm, *xm) - 2 * dot(*xp, *xm)) / (2 * pitch) ** 2
                    ad = (dot(*u, *xp) - dot(*u, *xm)) / (2 * pitch)
                    cx[z] = 1 / np.sqrt(np.maximum((dd - ad ** 2 / np.maximum(aa, 1e-300)) / sig ** 2, 1e-300))
                    zm = 2 * fz - z
                    if 0 <= zm < nz and zm != z:
                        v = (J0[:, zm], h[zm]); vv = dot(*v, *v); uv = dot(*u, *v)
                        dm[z] = np.sqrt(np.maximum(aa - uv ** 2 / np.maximum(vv, 1e-300), 0)) / sig
                rec = {"coder": coder, "rule": rule, "draw": draw, "k": k, "M": k * h.shape[-1], "dose": dose, "sigma": sig,
                       "crb_z_um": summarise(cz), "crb_x_um": summarise(cx), "dprime_mirror": summarise(dm),
                       "crb_z_by_plane": [float(np.nanmedian(cz[z])) if np.isfinite(cz[z]).any() else None for z in range(nz)],
                       "dprime_by_plane": [float(np.nanmedian(dm[z])) if np.isfinite(dm[z]).any() else None for z in range(nz)]}
                report["results"].append(rec)
                if dose == "frame":
                    np.savez_compressed(out / f"point_{coder}{a.tag}_{rule}{draw}_k{k}.npz", crb_z=cz, crb_x=cx, dprime_mirror=dm,
                                        lateral=np.array(lat), patterns=P)
                print(f"[point] {coder:7s} k={k:3d} {rule:6s}#{draw} {dose:5s} sigma_z {rec['crb_z_um']['median']:.4f} um "
                      f"(p90 {rec['crb_z_um']['p90']:.4f})  sigma_x {rec['crb_x_um']['median']:.4f}  "
                      f"d'_mirror {rec['dprime_mirror']['median']:.2f} (p10 {rec['dprime_mirror']['p10']:.2f})", flush=True)
        (out / f"point_information{a.tag}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[point] -> {out / f'point_information{a.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
