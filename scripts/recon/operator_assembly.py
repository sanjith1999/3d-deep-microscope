"""
Exact operator assembly from illumination and emitter footprints.

A coarse column is the sum over its fine patch of H2_t(f) * h(f), the illumination of pattern t at f times the
pooled detector image of an emitter at f.
"""
from __future__ import annotations
import copy

import torch
import torch.nn.functional as F

from coder import baselines
from coder.psf import conv_3D
from pipeline.forward import matched_dose


def illumination(meta, cfg, dev, chunk=32):
    c = copy.copy(cfg); c.n_layers = meta.get("physics", {}).get("n_layers", cfg.n_layers)
    c.d2nn_paraxial = meta.get("physics", {}).get("d2nn_paraxial", getattr(cfg, "d2nn_paraxial", False))
    T = meta["meas_shape"][0]; idx = meta.get("pattern_idx") or list(range(T))
    if meta["coder"] == "dmd":
        H2 = baselines.dmd_volume(c, seed=meta.get("seed", 0))[idx]
    else:
        with torch.no_grad():
            m = baselines.d2nn_model(c, seed=meta.get("seed", 0), coder_path=meta.get("designed") if meta["coder"] == "designed" else None)
            H2 = torch.cat([m(torch.tensor(idx[s:s + chunk], device=dev)).detach().cpu() for s in range(0, T, chunk)], 0)
    return matched_dose(H2.float(), cfg.dose_mean)                    # (T, nz, N, N) on CPU


def footprints(emPSF, nz, N, d, dev):
    m = nz // 2; c = N // 2; Nd = N // d
    X = torch.zeros(nz, nz, N, N, device=dev)
    for z in range(nz):
        X[z, z, c, c] = 1.0
    K = conv_3D(emPSF, X.to(torch.complex64), 2).real[:, m]          # (nz, N, N): centred-emitter images
    E = torch.zeros(nz, 3 * N + d, 3 * N + d, device=dev); E[:, N:2 * N, N:2 * N] = K
    S = F.avg_pool2d(E.unsqueeze(1), d, stride=1).squeeze(1)          # S[a, b] = mean of E over [a, a+d) x [b, b+d)
    x = torch.arange(N, device=dev); i = torch.arange(Nd, device=dev)
    a = (d * i.view(1, -1) - x.view(-1, 1) + c + N)                    # (N, Nd)
    return S[:, a[:, None, :, None], a[None, :, None, :]]             # (nz, N, N, Nd, Nd)


def assemble(H2, h, mode, patch=(2, 2, 2), dev="cuda"):
    """A (T*Nd*Nd, rz*rx*ry) for mode 'patch' or 'point'."""
    T, nz, N, _ = H2.shape; Nd = h.shape[-1]; pz, px, py = patch
    rz, rx, ry = nz // pz, N // px, N // py
    A = torch.empty(T * Nd * Nd, rz * rx * ry, dtype=torch.float32)
    members = [(dz, dx, dy) for dz in range(pz) for dx in range(px) for dy in range(py)] if mode == "patch" else [(0, 0, 0)]
    w = 1.0 if mode == "patch" else float(pz * px * py)
    for zc in range(rz):
        blk = torch.zeros(T, Nd * Nd, rx * ry, device=dev)
        for (dz, dx, dy) in members:
            z = zc * pz + dz
            Hm = H2[:, z, dx::px, dy::py].to(dev).reshape(T, -1)          # (T, rx*ry)
            hm = h[z, dx::px, dy::py].reshape(rx * ry, Nd * Nd)          # (rx*ry, Nd*Nd)
            blk += w * Hm[:, None, :] * hm.t()[None, :, :]
        A[:, zc * rx * ry:(zc + 1) * rx * ry] = blk.reshape(T * Nd * Nd, rx * ry).cpu()
    return A
