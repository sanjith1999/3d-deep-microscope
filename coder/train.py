"""
Training loop for the surrogate-objective coder design.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F

from coder.model import build_model
from coder.losses import (depth_decorrelation_loss, pattern_decorrelation_loss, pdf_match_loss,
                          contrast_loss, moment_loss, smoothness_loss, light_loss, axial_corr_loss)
from coder import targets
from coder.utils import seed_everything


def _plane_intensities(model, idx, bin_factor, lat_patch, pupil=True, dist_um=None):
    """Loss-plane intensity at the analysis (0.32 um) and recon lateral (0.64 um) resolutions."""
    I = model.plane_field(idx, pupil=pupil, dist_um=dist_um).abs().square().unsqueeze(1)
    Ia = F.avg_pool2d(I, bin_factor)                                # analysis grid
    Ir = F.avg_pool2d(Ia, lat_patch)                                # recon lateral grid
    return Ia.squeeze(1), Ir.squeeze(1)


@torch.no_grad()
def _axial_diagnostic(model, diag_idx, rfocus, patch):
    """+d/-d mirror correlation (depth_sym) of a few patterns relayed into 3D -- diagnostic only."""
    vol = model(diag_idx)
    _, sym = depth_decorrelation_loss(vol, rfocus, patch)
    return float(sym)


@torch.no_grad()
def _full_volume(model, chunk=32):
    parts = []
    for s in range(0, model.n_patterns, chunk):
        idx = torch.arange(s, min(s + chunk, model.n_patterns), device=model.inputs.device)
        parts.append(model(idx).cpu())
        torch.cuda.empty_cache()
    return torch.cat(parts, 0)


def train(cfg, out_dir: Path) -> dict:
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    init_ckpt = cfg.training.get("init_ckpt")
    if init_ckpt:                                         # start from an existing coder's phases
        with torch.no_grad():
            for p, s in zip(model.phases, torch.load(init_ckpt, map_location=device)["phases"]):
                p.copy_(s.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training["lr"])
    loss_mode = cfg.training.get("loss_mode", "stats")
    if loss_mode not in ("stats", "output"):
        raise ValueError(f"training.loss_mode must be 'stats' or 'output', got {loss_mode!r}")

    w = cfg.loss_weights
    target_name = cfg.target["name"]
    target_params = cfg.target.get("params", {})
    target_mean, target_var = targets.target_moments(target_name, **target_params)
    target_samples = targets.sample(target_name, cfg.training["mmd_samples"],
                                    device=device, seed=cfg.seed, **target_params)

    patch = tuple(cfg.recon_patch)
    rfocus = model.focus_idx // patch[0]                 # focal plane index on the recon grid
    T_all = model.n_patterns
    k_sub = cfg.training.get("pattern_subset", 0) or T_all
    loss_plane = cfg.training.get("loss_plane", "relay")
    if loss_plane not in ("relay", "object"):
        raise ValueError(f"training.loss_plane must be 'relay' or 'object', got {loss_plane!r}")
    loss_pupil = loss_plane == "relay"
    # loss_dist_um: None = the 4F object plane; a number = that fixed distance after the last layer;
    # [lo, hi] = a new uniformly random distance each step (shared by all patterns in the step).
    loss_dist = cfg.training.get("loss_dist_um")
    if isinstance(loss_dist, (list, tuple)):
        lo, hi = float(loss_dist[0]), float(loss_dist[1])
        draw_dist = lambda: lo + (hi - lo) * torch.rand(()).item()
    else:
        draw_dist = lambda: loss_dist
    n_diag = min(32, T_all)
    diag_idx = torch.linspace(0, T_all - 1, n_diag, device=device).round().long()

    target_contrast = float(target_var ** 0.5 / (target_mean + 1e-12))
    history = {k: [] for k in ("step", "total", "pattern_corr", "pattern_near", "pattern_nn",
                               "contrast", "contrast_loss", "pdf", "mean", "var", "smooth",
                               "diag_depth_sym", "light", "light_min", "c064", "axial_corr")}
    n_steps = cfg.training["num_steps"]
    t0 = time.time()
    c064_target = cfg.training.get("contrast064_target", 0.9)
    ax_dz = cfg.training.get("axial_dz_um", 1.28)

    for step in range(n_steps):
        opt.zero_grad()
        idx = None if k_sub >= T_all else torch.randperm(T_all, device=device)[:k_sub]
        dirs = model.scan_dirs if idx is None else model.scan_dirs[idx]
        d_step = draw_dist()

        if loss_mode == "output":
            E = model.plane_field(idx, pupil=True)
            I = E.abs().square()
            l_light, light, light_min = light_loss(I)
            Ir = F.avg_pool2d(I.unsqueeze(1), cfg.bin_factor * patch[1]).squeeze(1)      # 0.64 um
            l_c64, c64 = contrast_loss(Ir, c064_target)
            d_other = cfg.d2nn_out_dist_um + (ax_dz if torch.rand(()).item() < 0.5 else -ax_dz)
            Ib = model.plane_field(idx, pupil=True, dist_um=d_other).abs().square()
            Irb = F.avg_pool2d(Ib.unsqueeze(1), cfg.bin_factor * patch[1]).squeeze(1)
            l_ax = axial_corr_loss(Ir, Irb)
            total = (w.get("light", 1.0) * l_light + w.get("contrast064", 1.0) * l_c64
                     + w.get("axial", 1.0) * l_ax)
            total.backward()
            opt.step()
            if step % cfg.training["print_every"] == 0 or step == n_steps - 1:
                dsym = _axial_diagnostic(model, diag_idx, rfocus, patch)
                vals = {"step": step, "total": total.item(), "light": light.item(),
                        "light_min": light_min.item(), "c064": c64.item(), "axial_corr": l_ax.item(),
                        "diag_depth_sym": dsym}
                for k, v in vals.items():
                    history[k].append(v)
                print(f"[{step:5d}/{n_steps}] total={vals['total']:.4e} light={vals['light']:.3f} "
                      f"(min {vals['light_min']:.3f}) c064={vals['c064']:.3f} axial_corr={vals['axial_corr']:.3f} "
                      f"| diag depth_sym={dsym:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if cfg.training["checkpoint_every"] and step and step % cfg.training["checkpoint_every"] == 0:
                torch.save({"step": step, "phases": [p.detach().cpu() for p in model.phases],
                            "config": asdict(cfg)}, out_dir / f"ckpt_{step:05d}.pt")
            continue
        Ia, Ir = _plane_intensities(model, idx, cfg.bin_factor, patch[1], pupil=loss_pupil, dist_um=d_step)

        l_pat, l_near, l_nn = pattern_decorrelation_loss(Ir, dirs)
        l_con, con = contrast_loss(Ia, target_contrast)
        l_pdf = pdf_match_loss(Ia, target_samples, mmd_samples=cfg.training["mmd_samples"])
        l_mean, l_var = moment_loss(Ia, target_mean, target_var)
        l_smooth = smoothness_loss(torch.stack(model.phases))

        total = (w["pattern_decorr"] * l_pat + w.get("pattern_near", 0.0) * l_near
                 + w.get("contrast", 0.0) * l_con + w["pdf"] * l_pdf
                 + w["mean"] * l_mean + w["var"] * l_var + w["smooth"] * l_smooth)
        total.backward()
        opt.step()

        if step % cfg.training["print_every"] == 0 or step == n_steps - 1:
            dsym = _axial_diagnostic(model, diag_idx, rfocus, patch)
            vals = {"step": step, "total": total.item(), "pattern_corr": l_pat.item(),
                    "pattern_near": l_near.item(), "pattern_nn": l_nn.item(), "contrast": con.item(),
                    "contrast_loss": l_con.item(), "pdf": l_pdf.item(), "mean": l_mean.item(),
                    "var": l_var.item(), "smooth": l_smooth.item(), "diag_depth_sym": dsym}
            for k, v in vals.items():
                history[k].append(v)
            dtxt = "obj" if d_step is None else f"{d_step:.1f}um"
            print(f"[{step:5d}/{n_steps}] d={dtxt} total={vals['total']:.4e} corr={vals['pattern_corr']:.3f} "
                  f"near={vals['pattern_near']:.3f} nn={vals['pattern_nn']:.3f} "
                  f"contrast={vals['contrast']:.2f} pdf={vals['pdf']:.2e} | diag depth_sym={dsym:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        if cfg.training["checkpoint_every"] and step and \
                step % cfg.training["checkpoint_every"] == 0:
            torch.save({"step": step, "phases": [p.detach().cpu() for p in model.phases],
                        "config": asdict(cfg)}, out_dir / f"ckpt_{step:05d}.pt")

    wall = time.time() - t0
    torch.save({"phases": [p.detach().cpu() for p in model.phases], "config": asdict(cfg)},
               out_dir / "coder_final.pt")
    final_vol = _full_volume(model)
    torch.save(final_vol, out_dir / "speckle_volume.pt")

    last = {k: (v[-1] if v else None) for k, v in history.items()}
    manifest = {"config": asdict(cfg), "wall_seconds": wall, "focus_idx": model.focus_idx,
                "loss_mode": loss_mode, "init_ckpt": init_ckpt,
                "final_light": last["light"], "final_light_min": last["light_min"],
                "final_c064": last["c064"], "final_axial_corr": last["axial_corr"],
                "loss_plane": loss_plane, "loss_dist_um": loss_dist,
                "final_pattern_corr": last["pattern_corr"], "final_pattern_near": last["pattern_near"],
                "final_pattern_nn": last["pattern_nn"], "final_contrast": last["contrast"],
                "target_contrast": target_contrast,
                "final_pdf": last["pdf"], "final_diag_depth_sym": last["diag_depth_sym"],
                "device": str(device), "torch": torch.__version__}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "volume": final_vol, "model": model,
            "out_dir": out_dir, "manifest": manifest}
