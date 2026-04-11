"""
Surrogate design objectives computed on the illumination alone, without a reconstruction solver.

  Tier 1  axial decorrelation, weighting mirror pairs about the focal plane
  Tier 2  per-voxel intensity distribution and speckle contrast
  Tier 3  phase smoothness
"""
from __future__ import annotations
from typing import Sequence
import torch


# --- Tier 1: axial decorrelation (primary) ---------------------------------------------

def plane_matrix(vol: torch.Tensor) -> torch.Tensor:
    """Flatten (T,K,N,N) to (K, T*N*N) rows that are mean-removed and unit-norm."""
    K = vol.shape[1]
    P = vol.transpose(0, 1).reshape(K, -1)
    P = P - P.mean(dim=1, keepdim=True)
    return P / (P.norm(dim=1, keepdim=True) + 1e-12)


def _offdiag_sym(P: torch.Tensor, focus_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    K = P.shape[0]
    G = P @ P.t()

    off_mask = ~torch.eye(K, dtype=torch.bool, device=G.device)
    offdiag = G[off_mask].abs().mean()

    idx = torch.arange(K, device=G.device)
    mirror = 2 * focus_idx - idx                       # reflect about the focal plane
    sel = (mirror >= 0) & (mirror < K) & (mirror != idx)
    if sel.any():
        sym = G[idx[sel], mirror[sel]].abs().mean()
    else:
        sym = torch.zeros((), device=G.device)
    return offdiag, sym


def decorrelation_loss(vol: torch.Tensor, focus_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(offdiag, sym) on the RAW illumination planes."""
    return _offdiag_sym(plane_matrix(vol), focus_idx)


def recon_grid_pool(vol: torch.Tensor, patch: tuple[int, int, int]) -> torch.Tensor:
    """Pool illumination vol (T,K,N,N) to the reconstruction voxel grid."""
    pz, py, px = patch
    T, K, N, _ = vol.shape
    v = torch.nn.functional.avg_pool2d(vol.reshape(T * K, 1, N, N), kernel_size=(py, px))
    v = v.reshape(T, K, N // py, N // px)
    if pz > 1:
        Hs, Ws = v.shape[2], v.shape[3]
        v = torch.nn.functional.avg_pool1d(
            v.permute(0, 2, 3, 1).reshape(T * Hs * Ws, 1, K), kernel_size=pz)
        v = v.reshape(T, Hs, Ws, K // pz).permute(0, 3, 1, 2)
    return v                                                          # (T, K/pz, N/py, N/px)


def depth_gram_perpixel(vol_recon: torch.Tensor) -> torch.Tensor:
    """Averaged (Kr x Kr) depth correlation matrix over lateral pixels."""
    T, Kr, H, W = vol_recon.shape
    x = vol_recon.permute(2, 3, 1, 0).reshape(H * W, Kr, T)          # (P, z, T)
    x = x - x.mean(dim=2, keepdim=True)
    x = x / (x.norm(dim=2, keepdim=True) + 1e-12)
    return torch.matmul(x, x.transpose(1, 2)).mean(dim=0)            # (z, z)


def depth_decorrelation_loss(vol: torch.Tensor, focus_idx: int,
                             patch: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(offdiag, sym) of the recon-grid per-pixel depth-Gram -- the reconstruction-relevant term."""
    G = depth_gram_perpixel(recon_grid_pool(vol, patch))
    Kr = G.shape[0]
    off_mask = ~torch.eye(Kr, dtype=torch.bool, device=G.device)
    offdiag = G[off_mask].abs().mean()
    idx = torch.arange(Kr, device=G.device)
    mirror = 2 * focus_idx - idx
    sel = (mirror >= 0) & (mirror < Kr) & (mirror != idx)
    sym = G[idx[sel], mirror[sel]].abs().mean() if sel.any() else torch.zeros((), device=G.device)
    return offdiag, sym


def depth_coherence_score(vol: torch.Tensor, patch: tuple[int, int, int], min_sep: int = 1) -> float:
    """Max |correlation| over recon-grid depth pairs separated by >= min_sep planes (lower better)."""
    with torch.no_grad():
        G = depth_gram_perpixel(recon_grid_pool(vol, patch)).abs()
        Kr = G.shape[0]
        idx = torch.arange(Kr, device=G.device)
        mask = (idx.view(-1, 1) - idx.view(1, -1)).abs() >= max(1, min_sep)
        return float(G[mask].max().item()) if bool(mask.any()) else 0.0


def coherence_score(vol: torch.Tensor, min_sep: int = 1) -> float:
    """Max |correlation| over plane pairs separated by at least min_sep planes (lower better)."""
    with torch.no_grad():
        G = (plane_matrix(vol) @ plane_matrix(vol).t()).abs()
        K = G.shape[0]
        idx = torch.arange(K, device=G.device)
        mask = (idx.view(-1, 1) - idx.view(1, -1)).abs() >= max(1, min_sep)
        return float(G[mask].max().item()) if bool(mask.any()) else 0.0


def resolution_separation(dz_um: float, target_res_um: float = 1.0) -> int:
    """Plane separation (in indices) corresponding to a target axial resolution (~1 um)."""
    return max(1, round(target_res_um / dz_um)) if dz_um > 0 else 1


def coherence_profile(vol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Axial decorrelation curve: mean and max |correlation| vs plane separation."""
    with torch.no_grad():
        G = (plane_matrix(vol) @ plane_matrix(vol).t()).abs()
        K = G.shape[0]
        seps = list(range(1, K))
        means = torch.stack([torch.diagonal(G, offset=s).mean() for s in seps])
        maxes = torch.stack([torch.diagonal(G, offset=s).max() for s in seps])
        return torch.tensor(seps, dtype=torch.float), means, maxes


def pattern_correlation(x: torch.Tensor) -> torch.Tensor:
    """(T, T) |correlation| between patterns; x is (T, ...) intensities at the resolution of interest."""
    v = x.reshape(x.shape[0], -1)
    v = v - v.mean(dim=1, keepdim=True)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
    return (v @ v.t()).abs()


def pattern_decorrelation_loss(x: torch.Tensor, dirs: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (plain, near, nn) pattern-to-pattern |corr| for patterns x (T, ...)."""
    C = pattern_correlation(x)
    T = C.shape[0]
    off = ~torch.eye(T, dtype=torch.bool, device=C.device)
    D = torch.cdist(dirs, dirs)
    Dn = D.masked_fill(~off, float("inf"))
    sigma = Dn.min(dim=1).values.median().clamp_min(1e-6)
    G = torch.exp(-(D / sigma) ** 2) * off
    near = (G * C).sum() / (G.sum() + 1e-12)
    nn = C[torch.arange(T, device=C.device), Dn.argmin(dim=1)].mean()
    return C[off].mean(), near, nn


def light_loss(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kept = I.reshape(I.shape[0], -1).mean(dim=1)
    return -(kept + 1e-12).log().mean(), kept.mean(), kept.min()


def axial_corr_loss(Xa: torch.Tensor, Xb: torch.Tensor) -> torch.Tensor:
    a = Xa.reshape(Xa.shape[0], -1); b = Xb.reshape(Xb.shape[0], -1)
    a = a - a.mean(dim=0, keepdim=True); b = b - b.mean(dim=0, keepdim=True)
    c = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0) + 1e-12)
    return c.abs().mean()


# --- Tier 2: intensity distribution / contrast -----------------------------------------

def _normalise_intensity(vol: torch.Tensor) -> torch.Tensor:
    """Unit-mean scaling so only the PDF shape (contrast), not absolute power, is matched."""
    return vol / (vol.mean() + 1e-12)


def _pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.reshape(-1, 1) - b.reshape(1, -1)) ** 2


def mmd_rbf(x: torch.Tensor, y: torch.Tensor,
            bandwidths: Sequence[float] = (0.01, 0.05, 0.1, 0.5, 1.0)) -> torch.Tensor:
    """Multi-bandwidth RBF MMD^2 between two 1D sample sets."""
    dxx, dyy, dxy = _pairwise_sq_dists(x, x), _pairwise_sq_dists(y, y), _pairwise_sq_dists(x, y)
    kxx = kyy = kxy = 0.0
    for h in bandwidths:
        s = 2.0 * h * h
        kxx = kxx + torch.exp(-dxx / s)
        kyy = kyy + torch.exp(-dyy / s)
        kxy = kxy + torch.exp(-dxy / s)
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


def pdf_match_loss(vol: torch.Tensor, target_samples: torch.Tensor,
                   mmd_samples: int = 4096,
                   bandwidths: Sequence[float] = (0.01, 0.05, 0.1, 0.5, 1.0)) -> torch.Tensor:
    """Match the per-voxel intensity PDF of vol (T,K,N,N) to a target sample set via MMD."""
    flat = _normalise_intensity(vol).reshape(-1)
    if flat.numel() > mmd_samples:
        flat = flat[torch.randint(0, flat.numel(), (mmd_samples,), device=flat.device)]
    tgt = target_samples.to(flat.device)
    if tgt.numel() > mmd_samples:
        tgt = tgt[torch.randint(0, tgt.numel(), (mmd_samples,), device=tgt.device)]
    return mmd_rbf(flat, tgt, bandwidths)


def contrast_loss(x: torch.Tensor, target_contrast: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pattern speckle contrast (std/mean) pinned to the target's: returns (loss, mean contrast)."""
    v = x.reshape(x.shape[0], -1)
    c = v.std(dim=1) / (v.mean(dim=1) + 1e-12)
    return ((c - target_contrast) ** 2).mean(), c.mean()


def moment_loss(vol: torch.Tensor, target_mean: float,
                target_var: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the first two moments of the normalised intensity."""
    flat = _normalise_intensity(vol)
    return (flat.mean() - target_mean) ** 2, (flat.var() - target_var) ** 2


# --- Tier 3: regularisation ------------------------------------------------------------

def smoothness_loss(phases: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation on the phase masks (n_layer, N, N)."""
    dx = (phases[..., 1:, :] - phases[..., :-1, :]).abs().mean()
    dy = (phases[..., :, 1:] - phases[..., :, :-1]).abs().mean()
    return dx + dy
