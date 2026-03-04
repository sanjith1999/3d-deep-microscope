"""
Optical, numerical and training configuration shared by the coder and the reconstruction pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict, Any
import torch


@dataclass
class Config:
    # --- optics ---
    wavelength_um: float = 0.532
    na_ex: float = 0.8            # excitation NA (coherent, vectorial exPSF)
    na_em: float = 0.1            # emission NA (incoherent; intentional low-pass, not the coder's concern)
    n_index: float = 1.0

    nz: int = 32
    dz_um: float = 0.32

    n_fine: int = 384
    pitch_fine: float = 0.08
    bin_factor: int = 4          # n_fine / bin_factor = 96 analysis pixels, pitch_fine * bin = 0.32 um

    neuron_pitch_um: float = 0.32   # -> neuron_block = neuron_pitch_um / pitch_fine = 4
    n_layers: int = 2            # learnable phase masks (2-layer DOE; n_layers=1 models an SLM)
    delta_z_um: float = 3.3      # inter-layer distance
    d2nn_window: int = 2         # zero-pad factor for the D2NN AS layers
    exc_window: int = 2          # zero-pad factor for the exPSF spread convolution
    explicit_pupil: bool = True  # NA-0.8 aperture of the 4F relay applied to the D2NN output field
    d2nn_out_dist_um: float = 6.6
    d2nn_paraxial: bool = False   # True: Fresnel propagation in the element = the macro (intermediate-image) implementation

    # analysis-grid lateral pitch (pitch_fine * bin_factor); used by the emission PSF / detector.
    pitch_scalar: float = 0.32
    # --- scalar 2-layer coder (shelved): works on the analog grid directly, scalar depth spread ---
    n_scalar: int = 128          # analog grid (no fine/bin step for the scalar model)
    n_layers_scalar: int = 2     # learnable phase layers
    d2nn_window_scalar: int = 4  # D2NN AS window size
    exc_window_scalar: int = 8   # depth-spread window size

    # operator scale: normalise every coder's H2 to this mean so the operator magnitude matches the
    # level the recon rho/lmda are calibrated for (col-energy ~27; broad plateau ~20-100).
    dose_mean: float = 30.0

    n_alphas: int = 16
    n_thetas: int = 16           # n_patterns = n_alphas * n_thetas = 256

    recon_patch: tuple = (2, 2, 2)   # (z, y, x): (32,96,96) analysis grid -> (16,48,48) recon @ 0.64 um

    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "pattern_decorr": 4.0, "pattern_near": 4.0, "contrast": 2.0,
        "pdf": 1.0, "mean": 0.0, "var": 0.0, "smooth": 0.0,
        "light": 1.0, "contrast064": 2.0, "axial": 2.0})

    target: Dict[str, Any] = field(default_factory=lambda: {"name": "exponential", "params": {}})

    training: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 5e-3, "num_steps": 2000, "mmd_samples": 4096, "pattern_subset": 0,
        "loss_plane": "relay", "loss_dist_um": None, "loss_mode": "stats", "init_ckpt": None,
        "contrast064_target": 0.9, "axial_dz_um": 1.28, "print_every": 100, "checkpoint_every": 500})

    seed: int = 0
    out_dir: str = "results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_config() -> Config:
    return Config()
