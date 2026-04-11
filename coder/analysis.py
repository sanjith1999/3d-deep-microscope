"""
Speckle statistics of an illumination plane: grain size and power spectrum.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch


def autocorrelation_2d(intensity: np.ndarray) -> np.ndarray:
    centred = intensity - intensity.mean()
    power = np.abs(np.fft.fft2(centred)) ** 2
    ac = np.fft.fftshift(np.fft.ifft2(power).real)
    return ac / (ac.max() + 1e-12)


def radial_profile(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Radial average of a 2D array about its centre."""
    h, w = data.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - (w - 1) / 2.0) ** 2 + (y - (h - 1) / 2.0) ** 2).astype(int)
    profile = np.bincount(r.ravel(), data.ravel()) / np.maximum(1, np.bincount(r.ravel()))
    return np.arange(len(profile)), profile


def correlation_length(intensity: torch.Tensor, pixel_pitch: float) -> float:
    """Lateral speckle grain size: the radius where the autocorrelation falls to 1/e, in metres."""
    ac = autocorrelation_2d(intensity.detach().cpu().numpy().astype(np.float64))
    radii, profile = radial_profile(ac)
    idx = int(np.argmin(np.abs(profile - 1.0 / np.e)))
    return radii[idx] * pixel_pitch


def power_spectral_density(intensity: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D power spectral density and its radial average."""
    arr = intensity.detach().cpu().numpy().astype(np.float64)
    psd = np.fft.fftshift(np.abs(np.fft.fft2(arr)) ** 2)
    radii, psd_radial = radial_profile(psd)
    return radii, psd, psd_radial
