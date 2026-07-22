#!/usr/bin/env python3
"""
Fine-to-coarse operator mismatch ablation on identical measurements.

Arms: patch-based (basis-consistent) coarse operator, point-sampled coarse operator, and the device simulated
end-to-end on the reconstruction grid.

    python -u scripts/recon/mismatch_ablation.py all --op random1L_gap66 --label one_plate
"""
from __future__ import annotations
import argparse, csv, copy, hashlib, json, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT, _ROOT / "scripts" / "recon", _ROOT / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from configs.default import get_config
from coder import baselines
from coder.model import build_model
from pipeline.forward import ExportForward, emission_psf, matched_dose
from operator_assembly import illumination, footprints, assemble

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def illumination_coarse(meta, cfg, dev, f=2, chunk=32):
    if meta["coder"] == "dmd":
        raise SystemExit("coarse simulation is defined for D2NN coders")
    c = copy.copy(cfg); c.n_layers = meta.get("physics", {}).get("n_layers", cfg.n_layers)
    fine = baselines.d2nn_model(c, seed=meta.get("seed", 0), coder_path=meta.get("designed") if meta["coder"] == "designed" else None)
    ph = [p.detach()[::f, ::f].clone() for p in fine.phases]; del fine
    Na = c.n_fine // c.bin_factor
    c.pitch_fine = c.pitch_fine * c.bin_factor * f; c.n_fine = Na // f; c.bin_factor = 1
    c.neuron_pitch_um = c.pitch_fine; c.nz = c.nz // f; c.dz_um = c.dz_um * f; c.n_layers = len(ph)
    m = build_model(c)
    with torch.no_grad():
        for p, q in zip(m.phases, ph):
            p.copy_(q.to(p.device))
        T = meta["meas_shape"][0]; idx = meta.get("pattern_idx") or list(range(T))
        H2 = torch.cat([m(torch.tensor(idx[s:s + chunk], device=dev)).detach().cpu() for s in range(0, T, chunk)], 0)
    return matched_dose(H2.float(), cfg.dose_mean), c


def sha(t):
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


class Ablation:
    def __init__(self, a):
        self.a = a; self.cfg = get_config(); self.cfg.device = self.dev = a.device
        self.ops = _ROOT / "results/recon/ops"; self.meta = json.loads((self.ops / a.op / "meta.json").read_text())
        self.out = _ROOT / "results/paper/mismatch" / a.label; self.out.mkdir(parents=True, exist_ok=True)
        self.patch = tuple(self.meta["patch"]); self.d = self.meta["d_size"]

    def build(self):
        cfg, dev = self.cfg, self.dev
        t = time.time(); H2 = illumination(self.meta, cfg, dev)
        Nc, nz = H2.shape[2], H2.shape[1]
        emPSF = emission_psf(nz, Nc, cfg.dz_um, cfg.pitch_scalar, cfg.na_em, device=dev)
        h = footprints(emPSF, nz, Nc, self.d, dev)
        print(f"[mismatch] H2 {tuple(H2.shape)} + footprints in {time.time() - t:.0f}s", flush=True)
        checks = {"point_sample": "fine voxel (2zc, 2xc, 2yc): lower corner of the 2x2x2 patch, offset -0.5 fine voxel "
                  "(-0.16 um) per axis from the patch centre, weight 8 (mass of the patch)", "patch_weights": "1 per fine member"}
        for mode in ("patch", "point"):
            t = time.time(); A = assemble(H2, h, mode, self.patch, dev)
            torch.save(A, self.out / f"A_{mode}.pt")
            n = A.norm(dim=0)
            checks[f"A_{mode}"] = {"shape": list(A.shape), "secs": round(time.time() - t, 1), "sha256": sha(A),
                                   "col_norm": {"min": float(n.min()), "median": float(n.median()), "max": float(n.max())}}
            print(f"[mismatch] A_{mode} {tuple(A.shape)} in {time.time() - t:.0f}s", flush=True)
            if mode == "patch":
                Ab = torch.load(_ROOT / self.meta["A"], map_location="cpu", mmap=True)
                g = torch.Generator().manual_seed(0); cols = torch.randint(0, A.shape[1], (64,), generator=g)
                e = (A[:, cols] - Ab[:, cols].float()).norm() / Ab[:, cols].float().norm()
                checks["patch_vs_built_operator"] = {"columns": cols.tolist(), "rel_l2": float(e)}
                fwd = ExportForward(H2.to(dev), emPSF, patch=self.patch, d_size=self.d, chunk=16, dose_mean=None, device=dev)
                live = []
                for s in range(3):
                    x = torch.rand(*[nz // self.patch[0], Nc // self.patch[1], Nc // self.patch[2]], generator=torch.Generator().manual_seed(s))
                    y_live = fwd.forward(x.to(dev)).reshape(-1).cpu()
                    live.append(float((A @ x.reshape(-1) - y_live).norm() / y_live.norm()))
                checks["patch_vs_live_forward_rel_l2"] = live
                self.H2, self.emPSF, self.fwd = H2, emPSF, fwd
                print(f"[mismatch] A_patch vs built: {float(e):.2e}; vs live forward: {live}", flush=True)
            del A
        (self.out / "checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")

    def build_coarse(self):
        cfg, dev = self.cfg, self.dev; f = self.patch[1]
        t = time.time(); H2, c = illumination_coarse(self.meta, cfg, dev, f=f)
        nz, Nc = H2.shape[1], H2.shape[2]
        emPSF = emission_psf(nz, Nc, c.dz_um, c.pitch_fine, cfg.na_em, device=dev)
        h = footprints(emPSF, nz, Nc, self.d // f, dev)
        A = assemble(H2, h, "patch", (1, 1, 1), dev)
        Ap = torch.load(self.out / "A_patch.pt", mmap=True)
        one = torch.ones(A.shape[1]); scale = float((Ap @ one).sum() / (A @ one).sum()); A *= scale
        torch.save(A, self.out / "A_coarse.pt")
        g = torch.Generator().manual_seed(0); cols = torch.randint(0, A.shape[1], (64,), generator=g)
        col_rel = float((A[:, cols] - Ap[:, cols]).norm() / Ap[:, cols].norm())
        chk = {"coarse_sim": {"pitch_um": c.pitch_fine, "dz_um": c.dz_um, "grid": [nz, Nc, Nc], "d_size": self.d // f,
                              "neuron_phases": f"real plate, every {f}th 0.32 um neuron", "global_scale_vs_patch_uniform": scale,
                              "column_rel_l2_vs_patch_64cols": col_rel, "shape": list(A.shape), "sha256": sha(A),
                              "secs": round(time.time() - t, 1)}}
        p = self.out / "checks.json"; old = json.loads(p.read_text()) if p.exists() else {}
        p.write_text(json.dumps({**old, **chk}, indent=1), encoding="utf-8")
        print(f"[mismatch] A_coarse {tuple(A.shape)} scale {scale:.4g}; column rel L2 vs A_patch {col_rel:.3f} "
              f"({time.time() - t:.0f}s)", flush=True)

    def forward_residuals(self):
        Y = torch.load(self.ops / self.a.op / "Y.pt"); gts = torch.load(self.ops / "specimens.pt")["gt"]
        Ap = torch.load(self.out / "A_patch.pt", mmap=True)
        Ams = {m: (Ap if m == "patch" else torch.load(self.out / f"A_{m}.pt", mmap=True)) for m in self.a.modes}
        T, Nd1, Nd2 = self.meta["meas_shape"]
        rows = []
        tests = {s: (Y[s].float(), gts[s].reshape(-1).float()) for s in ("neural_cell", "beads")}
        g = gts["neural_cell"].float().reshape(-1)
        tests["control_patch_constant"] = (Ap @ g, g)
        for s, (y0, xr) in tests.items():
            sig = float(y0.mean()) * 10 ** (-self.a.q / 10)
            for mode, A in Ams.items():
                r = y0 - A @ xr
                per_frame = (r.reshape(T, -1).norm(dim=1) / y0.reshape(T, -1).norm(dim=1)).numpy()
                rows.append({"object": s, "operator": mode, "rel_l2_resid": float(r.norm() / y0.norm()),
                             "rms_resid_over_sigma_q": float(r.pow(2).mean().sqrt() / sig), "q_for_sigma": self.a.q,
                             "per_frame_rel_median": float(np.median(per_frame)), "per_frame_rel_max": float(per_frame.max())})
                np.save(self.out / f"resid_{s}_{mode}.npy", r.numpy().astype(np.float32))
                print(f"[mismatch] forward {s:24s} {mode:5s} rel {rows[-1]['rel_l2_resid']:.3e}  rms/sigma {rows[-1]['rms_resid_over_sigma_q']:.3f}", flush=True)
        with open(self.out / f"forward_residuals{self.a.tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def tune_coarse(self):
        import scico.numpy as snp
        from compression_subsets import Runner, TUNE_ON, rows_of
        from pipeline.reconstruct import run_pnp
        from pipeline.metrics import score
        r = Runner(argparse.Namespace(out=self.a.compression_out, snr=self.a.q, t_ref=256))
        T, Nd1, Nd2 = self.meta["meas_shape"]; rp = Nd1 * Nd2
        subs = json.loads((_ROOT / self.a.compression_out / "subsets.json").read_text())[self.a.op]
        A_all = snp.array(torch.load(self.out / "A_coarse.pt", mmap=True).numpy())
        p = self.out / "tune_coarse.json"; res = json.loads(p.read_text()) if p.exists() else {}
        log = self.out / "tune_coarse.jsonl"
        for preset, (spec, seed) in TUNE_ON.items():
            gt = r.lab.gt[spec]
            for k in self.a.ks:
                key = f"{preset}_{k}"
                if key in res:
                    continue
                pats = list(range(T)) if k >= T else subs[str(k)]["random"][0]
                rows = None if k >= T else rows_of(pats, rp)
                y, _ = r.lab.measure(self.a.op, spec, r.snr_for(k, T, "total"), seed)
                y = y if rows is None else y[rows]; A = A_all if rows is None else A_all[snp.array(rows)]
                base = r.tuned(preset, min(k, T)); tried = {}
                if log.exists():
                    for ln in log.read_text().splitlines():
                        q_ = json.loads(ln)
                        if q_["preset"] == preset and q_["k"] == k:
                            tried[(q_["l1s"], q_["rhos"])] = q_["psnr"]
                def run(l1s, rhos):
                    if (l1s, rhos) in tried:
                        return
                    st = {**base, "l1s": l1s, "rhos": rhos}; t = time.time()
                    est, _ = run_pnp(A, snp.array(y), obj_shape=gt.shape, gt=None, **r.prior(preset, st))
                    m = score(gt, np.asarray(est)); tried[(l1s, rhos)] = m["psnr"]
                    with open(log, "a") as f:
                        f.write(json.dumps({"preset": preset, "specimen": spec, "k": k, **st, "psnr": m["psnr"],
                                            "local_depth_err": m["local_depth_err"]}) + "\n")
                    print(f"[tune_coarse] {preset} k={k} l1x{l1s:g} rhox{rhos:g} PSNR {m['psnr']:.2f} depth {m['local_depth_err']:.2f} "
                          f"({time.time() - t:.0f}s)", flush=True)
                l1g = [base["l1s"]] if preset == "dncnn" else [1.0, 4.0, 16.0, 64.0]
                rhog = [0.0625, 0.25, 1.0, 4.0, 16.0] if preset == "dncnn" else [0.25, 1.0, 4.0]
                for l1s in l1g:
                    for rhos in rhog:
                        run(l1s, rhos)
                b = max(tried, key=tried.get)
                for rhos in (b[1] / 2, b[1] * 2):
                    run(b[0], rhos)
                for _ in range(4):                                 # extend past an edge until the best is interior
                    b = max(tried, key=tried.get)
                    R = sorted(r_ for l_, r_ in tried if l_ == b[0]); L = sorted({l_ for l_, _ in tried})
                    moved = False
                    if b[1] in (R[0], R[-1]):
                        run(b[0], b[1] * (4 if b[1] == R[-1] else 0.25)); moved = True
                    if preset == "l1" and b[0] in (L[0], L[-1]):
                        run(b[0] * (4 if b[0] == L[-1] else 0.25), b[1]); moved = True
                    if not moved:
                        break
                b = max(tried, key=tried.get)
                R = [r_ for l_, r_ in tried if l_ == b[0]]; L = [l_ for l_, _ in tried]
                res[key] = {**base, "l1s": b[0], "rhos": b[1], "psnr": tried[b],
                            "edge": b[1] in (min(R), max(R)) or (preset == "l1" and b[0] in (min(L), max(L)))}
                p.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))

    def recon(self):
        import scico.numpy as snp
        from compression_subsets import Runner, preset_of, rows_of
        from pipeline.reconstruct import run_pnp
        from pipeline.metrics import score, bead_report, local_depth_err
        from pipeline.metrics_paper import ssim3d, ssim3d_foreground, bead_localisation
        from pipeline.specimens import BEADS, mirror_plane
        r = Runner(argparse.Namespace(out=self.a.compression_out, snr=self.a.q, t_ref=256))
        T, Nd1, Nd2 = self.meta["meas_shape"]; rp = Nd1 * Nd2
        subs = json.loads((_ROOT / self.a.compression_out / "subsets.json").read_text())[self.a.op]
        As = {m: snp.array(torch.load(self.out / f"A_{m}.pt", mmap=True).numpy()) for m in sorted(set(self.a.modes) | {"patch"})}
        (self.out / "volumes").mkdir(exist_ok=True)
        out_rows = []; hist_all = {}
        for spec in ("neural_cell", "beads"):
            gt = r.lab.gt[spec]; preset = preset_of(spec)
            for k in self.a.ks:
                pats = list(range(T)) if k >= T else subs[str(k)]["random"][0]
                rows = None if k >= T else rows_of(pats, rp)
                st = r.tuned(preset, min(k, T)); prior_main = r.prior(preset, st); q = r.snr_for(k, T, "total")
                tc = json.loads((self.out / "tune_coarse.json").read_text()) if (self.out / "tune_coarse.json").exists() else {}
                for seed in self.a.seeds:
                    y, _ = r.lab.measure(self.a.op, spec, q, seed)           # one clean y, one noise draw, shared by both arms
                    y = y if rows is None else y[rows]
                    for mode in self.a.modes:
                        A = As[mode] if rows is None else As[mode][snp.array(rows)]
                        own = tc.get(f"{preset}_{k}") if mode == "coarse" else None
                        prior = r.prior(preset, {kk: own[kk] for kk in st}) if own else prior_main
                        t = time.time()
                        Apatch = As["patch"] if rows is None else As["patch"][snp.array(rows)]
                        ys = snp.array(y); yn = float(np.linalg.norm(y)); gn = float(np.linalg.norm(gt)); trace = []
                        def cb(sol, A=A, Apatch=Apatch, ys=ys, yn=yn, gn=gn, gt=gt, trace=trace):
                            x = sol.x.reshape(-1); xc = np.clip(np.asarray(sol.x), 0, 1)
                            trace.append({"fit_own": float(snp.linalg.norm(A @ x - ys)) / yn,
                                          "fit_common_fine_physics": float(snp.linalg.norm(Apatch @ x - ys)) / yn,
                                          "ref_err_rel_l2_clipped": float(np.linalg.norm(xc - gt)) / gn,
                                          "local_depth_err_planes_clipped": float(local_depth_err(gt, xc))})
                        est, _, hist = run_pnp(A, ys, obj_shape=gt.shape, gt=None, return_hist=True, callback=cb, **prior)
                        est = np.asarray(est)
                        res_arm = float(np.linalg.norm(np.asarray(A @ snp.array(est.reshape(-1))) - y) / np.linalg.norm(y))
                        res_common = float(np.linalg.norm(np.asarray(Apatch @ snp.array(est.reshape(-1))) - y) / np.linalg.norm(y))
                        m = score(gt, est)
                        row = {"op": self.a.op, "label": self.a.label, "specimen": spec, "k": k, "compression": 256 / k, "seed": seed,
                               "operator": mode, "q": q, "psnr": m["psnr"], "local_depth_err_planes": m["local_depth_err"],
                               "local_depth_err_um": m["local_depth_err"] * 0.64, "ssim3d": ssim3d(gt, est),
                               "ssim3d_fg": ssim3d_foreground(gt, est), "resid_under_own_operator": res_arm,
                               "resid_common_fine_physics": res_common, "secs": round(time.time() - t, 1),
                               "settings": json.dumps(prior)}
                        if spec == "beads":
                            br = bead_report(est, BEADS, mirror_plane); row["ghost"] = float(np.mean([b["ghost_ratio"] for b in br]))
                            row.update({f"bead_{kk}": v for kk, v in bead_localisation(est, BEADS, gt=gt).items()})
                        out_rows.append(row)
                        hk = f"{spec}_k{k}_s{seed}_{mode}"; hist_all[hk] = {}
                        try:
                            hist_all[hk] = {fld: [float(v) for v in getattr(hist, fld)] for fld in hist._fields}
                        except Exception:
                            pass
                        for fld in (trace[0] if trace else {}):
                            hist_all[hk][fld] = [t[fld] for t in trace]
                        row["cg_cap_hits"] = int(sum(1 for c in hist_all[hk].get("CG_It", []) if c >= 20))
                        np.savez_compressed(self.out / "volumes" / f"{spec}_k{k}_s{seed}_{mode}.npz", est=est.astype(np.float32), gt=gt)
                        print(f"[mismatch] {spec:11s} k={k:3d} s{seed} {mode:5s} PSNR {m['psnr']:.2f} depth {m['local_depth_err']:.2f} "
                              f"fgSSIM {row['ssim3d_fg']:.3f} res own {res_arm:.4f} common {res_common:.4f} ({row['secs']}s)", flush=True)
        keys = sorted({kk for rr in out_rows for kk in rr})
        with open(self.out / f"rows{self.a.tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out_rows)
        (self.out / f"histories{self.a.tag}.json").write_text(json.dumps(hist_all), encoding="utf-8")
        self.figure()

    def figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pipeline.specimens import BEADS, mirror_plane
        P = 0.64
        for spec in ("neural_cell", "beads"):
            vs = {}
            for mode in ("patch", "point", "coarse"):
                f = self.out / "volumes" / f"{spec}_k256_s0_{mode}.npz"
                if f.exists():
                    z = np.load(f); vs["reference"] = z["gt"]; vs[{"coarse": "coarse-simulated operator"}.get(mode, f"{mode}-based operator")] = z["est"]
            if not vs:
                continue
            order = ["reference", "coarse-simulated operator", "point-based operator", "patch-based operator"]; order = [o for o in order if o in vs]
            vmax = float(max(vs[o].max() for o in order))
            fig, ax = plt.subplots(2, len(order), figsize=(2.3 * len(order), 3.2), gridspec_kw={"height_ratios": [1, 16 / 48]})
            for j, o in enumerate(order):
                v = vs[o]
                ax[0, j].imshow(v.max(0), cmap="magma", vmin=0, vmax=vmax, extent=[0, 48 * P, 48 * P, 0])
                xz = v[:, :, 10] if spec == "beads" else v.max(2)
                ax[1, j].imshow(xz, cmap="magma", vmin=0, vmax=vmax, extent=[0, 48 * P, 16 * P, 0])
                ax[0, j].set_title(o, fontsize=7)
                for rr in range(2):
                    ax[rr, j].set_xticks([]); ax[rr, j].set_yticks([])
                if spec == "beads":
                    for (bz, bx, by) in BEADS:
                        if by == 10:
                            ax[1, j].plot((bx + .5) * P, (bz + .5) * P, "o", mfc="none", mec="c", ms=3, mew=.5)
                            ax[1, j].plot((bx + .5) * P, (mirror_plane(bz) + .5) * P, "x", color="r", ms=3, mew=.5)
            ax[0, 0].set_ylabel("xy MIP", fontsize=6); ax[1, 0].set_ylabel("xz MIP" if spec != "beads" else "xz slice y=10", fontsize=6)
            fig.suptitle(f"{spec}: full scan, noise seed 0, identical measurements; shared limits [0, {vmax:.2f}]; {self.a.label}", fontsize=6)
            for ext in ("pdf", "svg", "png"):
                fig.savefig(self.out / f"fig_{spec}.{ext}", dpi=600, bbox_inches="tight")
            plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch- vs point-sampled coarse operator ablation.")
    ap.add_argument("stage", choices=["build", "build_coarse", "tune_coarse", "forward", "recon", "figure", "all"])
    ap.add_argument("--op", default="random_gap66")
    ap.add_argument("--compression_out", default="results/recon/compression_q15")
    ap.add_argument("--label", default="pilot_2layer")
    ap.add_argument("--q", type=float, default=15.0)
    ap.add_argument("--ks", default="256,64"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", default="patch,point", help="operators to evaluate (patch, point, coarse)")
    a = ap.parse_args(); a.modes = a.modes.split(","); a.tag = "" if a.modes == ["patch", "point"] else "_" + "_".join(a.modes); a.ks = [int(x) for x in a.ks.split(",")]; a.seeds = [int(x) for x in a.seeds.split(",")]
    ab = Ablation(a)
    if a.stage in ("build", "all"):
        ab.build()
    if a.stage == "build_coarse":
        ab.build_coarse()
    if a.stage == "tune_coarse":
        ab.tune_coarse()
    if a.stage == "figure":
        ab.figure()
    if a.stage in ("forward", "all"):
        ab.forward_residuals()
    if a.stage in ("recon", "all"):
        ab.recon()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
