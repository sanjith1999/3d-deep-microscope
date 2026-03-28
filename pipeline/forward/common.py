"""
Optics and object helpers shared by the forward models: emission PSF, dose matching, pooling, phantoms.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from coder.psf import psf_model


def matched_dose(H2: torch.Tensor, target_mean: float = 1.0) -> torch.Tensor:
    return H2 / (H2.mean() + 1e-12) * target_mean


def emission_psf(nz: int, N: int, dz_um: float, pitch_um: float, na_em: float,
                 n_index: float = 1.0, lambda_um: float = 0.532, device: str = "cuda") -> torch.Tensor:
    """Incoherent emission intensity PSF (1, nz, N, N): |E_em|^2 summed over components, at em_NA."""
    em = psf_model(na_em, n_index, lambda_um, pitch_um, pitch_um, dz_um, N, N, nz)()
    return em.detach().permute(0, 3, 1, 2).abs().square().sum(dim=0).unsqueeze(0).to(device)


def spherical_object(nz: int, N: int, dz_um: float, pitch_um: float, radius_um: float = 3.0,
                     device: str = "cuda") -> torch.Tensor:
    """A solid sphere on the analog grid (nz, N, N), values in {0,1}."""
    z = (torch.arange(nz) - nz // 2).float() * dz_um
    xy = (torch.arange(N) - N // 2).float() * pitch_um
    zz, xx, yy = torch.meshgrid(z, xy, xy, indexing="ij")
    return (torch.sqrt(zz ** 2 + xx ** 2 + yy ** 2) <= radius_um).float().to(device)


def reduce_object(X_analog: torch.Tensor, patch=(4, 4, 4)) -> torch.Tensor:
    """Average-pool an analog object (nz,N,N) to the recon grid (rz,rx,ry)."""
    return F.avg_pool3d(X_analog.unsqueeze(0).unsqueeze(0), patch).squeeze()


def bead_phantom(nz: int, N: int, beads, patch=(4, 4, 4), radius_um: float = 1.0,
                 dz_um: float = 0.25, pitch_um: float = 0.25, device: str = "cuda") -> torch.Tensor:
    """Point beads at KNOWN recon positions, rendered as small spheres on the analog grid."""
    pz, px, py = patch
    z = torch.arange(nz).float()
    xy = torch.arange(N).float()
    zz, xx, yy = torch.meshgrid(z, xy, xy, indexing="ij")
    X = torch.zeros(nz, N, N)
    for (rz, rx, ry) in beads:
        cz, cx, cy = rz * pz + pz / 2.0, rx * px + px / 2.0, ry * py + py / 2.0
        d = torch.sqrt(((zz - cz) * dz_um) ** 2 + ((xx - cx) * pitch_um) ** 2 + ((yy - cy) * pitch_um) ** 2)
        X = torch.maximum(X, (d <= radius_um).float())
    return X.to(device)


def add_noise_snr(Y: torch.Tensor, snr_db: float, seed: int = 0) -> torch.Tensor:
    """Add Gaussian noise at a target SNR."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    power = Y.mean().item()
    noise_power = power / (10 ** (snr_db / 10))
    noise = torch.randn(Y.shape, generator=g).to(Y.device) * noise_power
    return Y + noise
