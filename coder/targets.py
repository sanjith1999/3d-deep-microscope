"""
Target intensity distributions for the PDF-matching objective (bimodal, exponential, soft binary).
"""
from __future__ import annotations
from typing import Optional, Sequence
import torch


def _unit_mean(x: torch.Tensor) -> torch.Tensor:
    return x / (x.mean() + 1e-12)


def _generator(device, seed: Optional[int]):
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def exponential(n: int, device=None, seed: Optional[int] = None) -> torch.Tensor:
    """Negative-exponential PDF: the marginal of fully-developed speckle (contrast = 1)."""
    u = torch.rand(n, device=device, generator=_generator(device, seed)).clamp_min(1e-12)
    return _unit_mean(-torch.log(u))


def soft_binary(n: int, device=None, p_on: float = 0.5, sharpness: float = 0.05,
                seed: Optional[int] = None) -> torch.Tensor:
    gen = _generator(device, seed)
    on = (torch.rand(n, device=device, generator=gen) < p_on).float()
    noise = sharpness * torch.randn(n, device=device, generator=gen)
    return _unit_mean((on + noise).clamp_min(0.0))


def bimodal(n: int, device=None, seed: Optional[int] = None,
            weights: Sequence[float] = (0.5, 0.5), means: Sequence[float] = (0.5, 1.5),
            stds: Sequence[float] = (0.1, 0.2)) -> torch.Tensor:
    """Bimodal Gaussian mixture."""
    gen = _generator(device, seed)
    w = torch.tensor(weights, dtype=torch.float)
    w = w / w.sum()
    means_t = torch.tensor(means, dtype=torch.float)
    stds_t = torch.tensor(stds, dtype=torch.float)
    comp = torch.multinomial(w, n, replacement=True, generator=gen)
    s = torch.randn(n, device=device, generator=gen) * stds_t[comp] + means_t[comp]
    return _unit_mean(s.clamp_min(0.0))


_TARGETS = {"exponential": exponential, "soft_binary": soft_binary, "bimodal": bimodal}


def sample(name: str, n: int, device=None, seed: Optional[int] = None, **kw) -> torch.Tensor:
    """Dispatch by target name: 'exponential' | 'soft_binary' | 'bimodal'."""
    name = name.lower()
    if name not in _TARGETS:
        raise ValueError(f"unknown target distribution: {name!r}")
    out = _TARGETS[name](n, device=torch.device("cpu"), seed=seed, **kw)
    return out.to(device) if device is not None else out


def target_moments(name: str, n: int = 100_000, **kw) -> tuple[float, float]:
    """Empirical (mean, var) of a target at unit mean -- for the moment-matching loss."""
    s = sample(name, n, device=torch.device("cpu"), seed=0, **kw)
    return float(s.mean()), float(s.var())
