"""
Illumination volumes of the coders compared in this project.

  dmd          binary amplitude patterns in focus at the 4F object plane (axially symmetric illumination)
  random_d2nn  the diffractive plate with random phases
  designed     the diffractive plate with trained phases

All coders share the same vectorial excitation relay, so only the object-plane field differs.
"""
from __future__ import annotations
import copy
from pathlib import Path

import torch

from coder.model import DiffractiveCoder, build_model
from coder.excitation import VectorialExcitation
from coder.losses import (plane_matrix, decorrelation_loss, coherence_score,
                          coherence_profile, resolution_separation)


def _excitation(cfg) -> VectorialExcitation:
    return VectorialExcitation(cfg.n_fine, cfg.pitch_fine, cfg.dz_um, cfg.nz, cfg.na_ex,
                               n_index=cfg.n_index, lambda_um=cfg.wavelength_um,
                               bin_factor=cfg.bin_factor, w=cfg.exc_window, device=cfg.device,
                               checkpoint_planes=False)


@torch.no_grad()
def dmd_fields(cfg, seed: int = 0) -> torch.Tensor:
    T = cfg.n_alphas * cfg.n_thetas
    Nc, b = cfg.n_fine // cfg.bin_factor, cfg.bin_factor
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coarse = (torch.rand(T, Nc, Nc, generator=gen) < 0.5).float()
    return coarse.repeat_interleave(b, 1).repeat_interleave(b, 2).to(cfg.device).to(torch.complex64)


@torch.no_grad()
def dmd_volume(cfg, seed: int = 0) -> torch.Tensor:
    """The DMD patterns spread into 3D by the same vectorial 4F relay (exPSF)."""
    return _excitation(cfg).spread(dmd_fields(cfg, seed)).detach()


@torch.no_grad()
def d2nn_model(cfg, seed: int = 0, coder_path: str | Path | None = None) -> DiffractiveCoder:
    if coder_path is None:
        model = build_model(cfg)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for p in model.phases:
            p.copy_((2 * torch.pi * torch.rand(p.shape, generator=gen)).to(p.device))
        return model
    ckpt = torch.load(coder_path, map_location=cfg.device)
    saved = ckpt["phases"] if isinstance(ckpt, dict) and "phases" in ckpt else ckpt
    if len(saved) != cfg.n_layers:
        cfg = copy.copy(cfg); cfg.n_layers = len(saved)
    model = build_model(cfg)
    for p, s in zip(model.phases, saved):
        p.copy_(s.to(p.device))
    return model


@torch.no_grad()
def random_d2nn_volume(cfg, seed: int = 0) -> torch.Tensor:
    """The D2NN with random (untrained) phase masks."""
    return d2nn_model(cfg, seed=seed)().detach()


@torch.no_grad()
def designed_volume(cfg, coder_path: str | Path) -> torch.Tensor:
    """The trained coder: load the saved phase masks into a fresh DiffractiveCoder."""
    return d2nn_model(cfg, coder_path=coder_path)().detach()


@torch.no_grad()
def relay_light(cfg, coder: str, seed: int = 0, coder_path=None, chunk: int = 32) -> float:
    if coder == "dmd":
        f = dmd_fields(cfg, seed)
        fr = torch.fft.fftfreq(cfg.n_fine, d=cfg.pitch_fine, device=cfg.device)
        pupil = (torch.sqrt(fr.view(-1, 1) ** 2 + fr.view(1, -1) ** 2) <= cfg.na_ex / cfg.wavelength_um)
        return float(torch.fft.ifft2(torch.fft.fft2(f) * pupil).abs().square().mean())
    m = d2nn_model(cfg, seed=seed, coder_path=coder_path if coder == "designed" else None)
    T = m.n_patterns
    tot = sum(float(m.plane_field(torch.arange(s, min(s + chunk, T), device=cfg.device)).abs().square().sum())
              for s in range(0, T, chunk))
    return tot / (T * cfg.n_fine ** 2)


@torch.no_grad()
def score_volume(vol: torch.Tensor, focus_idx: int, dz_um: float) -> dict:
    """Separation-aware coherence metrics + correlation matrix and profile (small/JSON-friendly)."""
    offdiag, sym = decorrelation_loss(vol, focus_idx)
    res_sep = resolution_separation(dz_um)
    coh = coherence_score(vol)
    coh_res = coherence_score(vol, min_sep=res_sep)
    seps, mean_c, max_c = coherence_profile(vol)
    gram = (plane_matrix(vol) @ plane_matrix(vol).t()).abs().cpu().numpy()
    return {"offdiag": float(offdiag), "sym": float(sym), "coherence": float(coh),
            "coherence_res": float(coh_res), "res_sep": int(res_sep), "gram": gram,
            "seps": seps.cpu().numpy(), "mean_c": mean_c.cpu().numpy(), "max_c": max_c.cpu().numpy()}
