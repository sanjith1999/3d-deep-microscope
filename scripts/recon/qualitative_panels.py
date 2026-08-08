#!/usr/bin/env python3
"""
Qualitative panels from saved reference and reconstruction volumes with shared intensity limits.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from pipeline.specimens import BEADS, COHORT, mirror_plane

PITCH = 0.64
BEAD_Y = 10


def find(d, op, k, specimen, seed, rule, draw):
    subs = json.loads((d / "subsets.json").read_text())[op][str(k)]
    pats = subs["all" if "all" in subs else rule][0 if "all" in subs else draw]
    h = hashlib.sha1(json.dumps(sorted(int(p) for p in pats)).encode()).hexdigest()[:10]
    hits = sorted((d / "volumes").glob(f"{op}_k{k}_{h}_{specimen}_s{seed}_*.npz"))
    return (np.load(hits[0])["est"].astype(np.float32), hits[0].name, pats) if hits else (None, None, pats)


def metrics(d, fname):
    p = d / "volume_metrics.csv"
    if p.exists() and fname:
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            if r["file"] == fname:
                return {k: r[k] for k in ("ssim3d", "ssim3d_fg", "nrmse", "fidelity_res_axial_um", "fidelity_res_lateral_um") if k in r}
    return {}


def depth_rgb(v, vmax, cmap):
    z = v.argmax(0) / (v.shape[0] - 1); a = np.clip(v.max(0) / vmax, 0, 1)
    return cmap(z)[..., :3] * a[..., None]


def scalebar(ax, um=5.0, y_frac=0.92, color="w"):
    x0 = 2; ax.plot([x0, x0 + um / PITCH], [ax.get_ylim()[0] * y_frac] * 2, color=color, lw=1.5, solid_capstyle="butt")


def main() -> int:
    ap = argparse.ArgumentParser(description="Qualitative reconstruction panels from saved volumes.")
    ap.add_argument("--dir", default="results/recon/compression_q15")
    ap.add_argument("--op", default="random_gap66")
    ap.add_argument("--label", default="preview_2layer")
    ap.add_argument("--k", type=int, default=64, help="limited-frame column")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--specimens", default="neural_cell,beads", help="specimens to panel (neuronal names neural_cell[_p<i>])")
    ap.add_argument("--ks", default=None, help="frame-count columns after the reference, e.g. 256,64,32 (default 256,<k>)")
    ap.add_argument("--no_contact", action="store_true")
    a = ap.parse_args(); a.specimens = a.specimens.split(",")
    a.ks = [int(x) for x in a.ks.split(",")] if a.ks else [256, a.k]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})
    d = _ROOT / a.dir; gts = {k: v.numpy() for k, v in torch.load(_ROOT / "results/recon/ops/specimens.pt")["gt"].items()}
    meta = json.loads((_ROOT / "results/recon/ops" / a.op / "meta.json").read_text())
    q = json.loads((d / "tune.json").read_text()) if (d / "tune.json").exists() else {}
    base = _ROOT / "results/paper/package/qualitative" / a.label
    cm = plt.get_cmap("viridis")
    rz, rx, ry = 16, 48, 48; ext_xy = [0, rx * PITCH, ry * PITCH, 0]; ext_xz = [0, rx * PITCH, rz * PITCH, 0]

    runs = [json.loads(l) for f in sorted(d.glob("runs*.jsonl")) for l in open(f) if l.strip()]
    Yops = torch.load(_ROOT / "results/recon/ops" / a.op / "Y.pt")

    def run_metrics(spec, k):
        """PSNR / depth error / settings of the DISPLAYED run (same specimen, k, noise seed, draw)."""
        hit = [r for r in runs if r["specimen"] == spec and r["k"] == k and r["seed"] == a.seed
               and r.get("draw", 0) == (a.draw if k < 256 else 0)]
        if not hit:
            return {}
        r = hit[0]
        return {"psnr_db": r["psnr"], "local_depth_err_planes": r["local_depth_err"], "local_depth_err_um": r["local_depth_err"] * PITCH,
                "q_k": r["snr"], "settings_scales": {kk: r[kk] for kk in ("l1s", "rhos", "max_iter") if kk in r}}

    def sigma(spec, k):
        m = float(Yops[spec].float().mean()); qk = 15 + 5 * np.log10(256 / k)
        return {"mean_y0_full_256_frame_stack": m, "q_k": float(qk), "sigma": m * 10 ** (-qk / 10)}

    def spec_provenance(spec):
        if spec.startswith("neural_cell"):
            return ("Reference = microscopy-derived crop (mouse pyramidal neuron data from DEEP2, PS_SOM_mice_20190317.mat "
                    "Data/cell[3,0]); one parent stack, crop coordinates unknown, overlap between crops possible; 0.32 um voxel "
                    "scale ASSIGNED for simulation. Measurements and reconstructions are simulated. See DATA_PROVENANCE.md.")
        return "Synthetic bead phantom (spheres r = 0.8 um at recon-voxel centres, mirror pairs about focus); simulated."

    def provenance(spec, k, fname, pats):
        return {"operator": a.op, "layers": None if meta["coder"] == "dmd" else meta.get("physics", {}).get("n_layers"), "coder": meta["coder"],
                "element_seed": meta.get("seed"), "propagation": "n/a (DMD)" if meta["coder"] == "dmd" else ("paraxial (Fresnel)" if meta.get("physics", {}).get("d2nn_paraxial") else "exact (non-paraxial angular spectrum)"), "k": k,
                "subset": "all" if k == 256 else f"random draw {a.draw}", "pattern_indices": pats if k < 256 else "all 256",
                "noise_seed": a.seed,
                "noise_rule": "sigma = mean(y0 over this encoder's FULL 256-frame clean stack) * 10^(-q_k/10), q_k = 15 + 5 log10(256/k); "
                              "rows selected after noise scaling; Gaussian, not a calibrated photon count",
                "noise_sigma": sigma(spec, k), "file": fname, "tuned_settings": q,
                "metrics_this_run": {**metrics(d, fname), **run_metrics(spec, k)}}

    for spec in a.specimens:
        out = base / spec; out.mkdir(parents=True, exist_ok=True)
        gt = gts[spec]; cols = [("reference", gt, None, None)]
        for k in a.ks:
            est, fname, pats = find(d, a.op, k, spec, a.seed, "random", a.draw)
            if est is not None:
                cols.append((f"{'full scan (256 frames)' if k == 256 else f'{k} frames ({256 // k}x fewer)'}", est, fname, (k, pats)))
        vmax = float(max(c[1].max() for c in cols))
        n = len(cols)
        fig, ax = plt.subplots(3, n, figsize=(2.3 * n, 4.6), gridspec_kw={"height_ratios": [1, 1, rz / rx * 1.0]})
        arrays = {}
        for j, (title, v, fname, _) in enumerate(cols):
            ax[0, j].imshow(depth_rgb(v, vmax, cm), extent=ext_xy)
            ax[1, j].imshow(v.max(0), cmap="magma", vmin=0, vmax=vmax, extent=ext_xy)
            xz = v[:, :, BEAD_Y] if spec == "beads" else v.max(2)   # beads: slice through one bead column (ghosts visible)
            ax[2, j].imshow(xz, cmap="magma", vmin=0, vmax=vmax, extent=ext_xz, aspect="equal")
            arrays.update({f"{j}_{title}__xy_mip": v.max(0), f"{j}_{title}__xz_mip_over_y": v.max(2),
                           f"{j}_{title}__depth_index": v.argmax(0), f"{j}_{title}__volume_clipped": v})
            if spec == "beads":                                   # the bead figure DISPLAYS a slice, not the MIP
                arrays[f"{j}_{title}__xz_slice_y{BEAD_Y}_displayed"] = v[:, :, BEAD_Y]
            ax[0, j].set_title(title, fontsize=7)
            for r in range(3):
                ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if spec == "beads":
                for (bz, bx, by) in BEADS:
                    ax[0, j].plot((by + .5) * PITCH, (bx + .5) * PITCH, "o", mfc="none", mec="w", ms=4, mew=0.5)
                    if by == BEAD_Y:                                   # beads in the sliced column
                        ax[2, j].plot((bx + .5) * PITCH, (bz + .5) * PITCH, "o", mfc="none", mec="c", ms=3, mew=0.5)
                        ax[2, j].plot((bx + .5) * PITCH, (mirror_plane(bz) + .5) * PITCH, "x", color="r", ms=3, mew=0.5)
        ax[0, 0].set_ylabel("xy MIP, colour = depth", fontsize=6); ax[1, 0].set_ylabel("xy MIP (linear)", fontsize=6)
        ax[2, 0].set_ylabel("xz MIP (z down)" if spec != "beads" else f"xz slice, bead column y={BEAD_Y}", fontsize=6)
        for r in (1, 2):
            ax[r, 0].plot([1, 1 + 5], [ext_xy[2] * 0.93 if r == 1 else ext_xz[2] * 0.85] * 2, color="w", lw=1.2)
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.92, 0.66, 0.012, 0.22])            # own axis: does not shrink any panel
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, rz * PITCH)); cb = fig.colorbar(sm, cax=cax)
        cb.set_label("depth (µm)", fontsize=6); cb.ax.tick_params(labelsize=5)
        dev_label = ("in-focus DMD baseline" if meta["coder"] == "dmd" else
                     f"{meta.get('physics', {}).get('n_layers')}-layer element" +
                     (", paraxial propagation" if meta.get("physics", {}).get("d2nn_paraxial") else ""))
        spec_title = ("synthetic bead phantom" if not spec.startswith("neural_cell") else
                      "neuronal fluorescence volume" + ("" if spec == "neural_cell" else " (subvolume " + spec.rsplit("_p", 1)[-1] + ")"))
        fig.suptitle(f"{spec_title} — simulated "
                     f"measurement and reconstruction; {dev_label} ({a.op}); "
                     f"5 µm bar; shared limits [0, {vmax:.2f}]", fontsize=6)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out / f"panel_{spec}.{ext}", dpi=600)
        plt.close(fig)
        man = {"specimen": spec, "provenance": spec_provenance(spec),
               "arrays": {"__xy_mip": "max over z (displayed rows 1-2)",
                          "__xz_mip_over_y": "max over y" + (" (displayed row 3)" if spec != "beads" else " (NOT displayed for beads)"),
                          "__xz_slice_y%d_displayed" % BEAD_Y: "beads only: the displayed row-3 slice at y index %d" % BEAD_Y,
                          "__depth_index": "argmax over z (depth colour)", "__volume_clipped": "full (16,48,48) volume, final [0,1] clip",
                          "volume order": "(z, x, y), 0.64 um voxels"},
               "display": {"intensity_limits": [0, vmax], "palette": "magma (linear)", "depth_palette": "viridis",
                                             "depth_overview": "RGB = viridis(depth of max)/(nz-1) x MIP/vmax", "pitch_um": PITCH,
                                             "aspect": "equal physical units (no stretch)", "crop": "full field",
                                             "xz_view": "max over y" if spec != "beads" else f"slice at y index {BEAD_Y} (pre-selected bead column); o = true depth, x = mirror depth"},
               "columns": [{"title": t, **(provenance(spec, kp[0], f, kp[1]) if kp else {"reference": "2x2x2 average-pooled fine object"})}
                           for (t, _, f, kp) in cols]}
        if spec == "beads":
            bz, bx, by = BEADS[0]; prof = {t: v[:, max(bx - 1, 0):bx + 2, max(by - 1, 0):by + 2].sum((1, 2)) for (t, v, _, _) in cols}
            fig, axp = plt.subplots(figsize=(3.0, 1.8))
            for t, p in prof.items():
                axp.plot((np.arange(rz) + .5) * PITCH, p, lw=1, label=t)
            axp.axvline((bz + .5) * PITCH, c="k", lw=0.6, ls="--"); axp.axvline((mirror_plane(bz) + .5) * PITCH, c="r", lw=0.6, ls=":")
            axp.set(xlabel="depth (µm)", ylabel="summed intensity (3x3 column)"); axp.legend(fontsize=5, frameon=False)
            axp.set_title(f"pre-selected bead 0 at plane {bz} (dashed); mirror plane {mirror_plane(bz)} (dotted)", fontsize=6)
            for ext in ("pdf", "svg", "png"):
                fig.savefig(out / f"profile_bead0.{ext}", dpi=600, bbox_inches="tight")
            plt.close(fig); arrays.update({f"profile_{t}": p for t, p in prof.items()})
            man["profile"] = {"bead": BEADS[0], "mirror_plane": mirror_plane(bz), "window": "3x3 lateral column, summed",
                              "selection": "bead 0 of BEADS, fixed before inspection"}
        np.savez_compressed(out / f"arrays_{spec}.npz", **arrays)
        (out / f"manifest_{spec}.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")

    if a.no_contact:
        return 0
    # contact sheet of the cohort at full scan
    out = base / "contact"; out.mkdir(parents=True, exist_ok=True)
    specs = ["neural_cell"] + COHORT; rows = []
    fig, ax = plt.subplots(2, len(specs), figsize=(1.6 * len(specs), 2.6))
    for j, s in enumerate(specs):
        est, fname, _ = find(d, a.op, 256, s, a.seed if s == "neural_cell" else 0, "random", 0)
        g = gts.get(s)
        if est is None or g is None:
            continue
        vm = float(max(g.max(), est.max()))
        ax[0, j].imshow(np.concatenate([g.max(0), est.max(0)], 1), cmap="magma", vmin=0, vmax=vm)
        ax[1, j].imshow(np.concatenate([g.max(2), est.max(2)], 1), cmap="magma", vmin=0, vmax=vm)
        m = metrics(d, fname); fill = float((g > 0.1 * g.max()).mean())
        ax[0, j].set_title(f"{s.replace('neural_cell', 'n')}\nfill {100 * fill:.1f}%  fgSSIM {float(m.get('ssim3d_fg', 'nan')):.2f}", fontsize=5)
        for r in range(2):
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        rows.append({"specimen": s, "fill": fill, "file": fname, **m})
    ax[0, 0].set_ylabel("xy MIP: ref | recon", fontsize=5); ax[1, 0].set_ylabel("xz MIP: ref | recon", fontsize=5)
    fig.suptitle(f"neuronal subvolumes, full scan, {a.op}; per-column shared limits (ref and recon of one specimen)", fontsize=6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"contact_sheet.{ext}", dpi=400, bbox_inches="tight")
    (out / "contact_sheet.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"[qualitative] -> {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
