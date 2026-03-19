"""
DiffractiveCoder: angle-scanned input -> D2NN -> NA-limited relay -> 3D excitation volume.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from coder.diffractive import D2NNStack, angle_scan_inputs, angle_scan_directions
from coder.excitation import VectorialExcitation


class DiffractiveCoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dev = cfg.device
        neuron_block = round(cfg.neuron_pitch_um / cfg.pitch_fine)
        self.d2nn = D2NNStack(cfg.n_fine, cfg.pitch_fine, cfg.wavelength_um, cfg.n_layers,
                              cfg.delta_z_um, w=cfg.d2nn_window, neuron_block=neuron_block,
                              out_dz_um=getattr(cfg, "d2nn_out_dist_um", None), device=dev, paraxial=getattr(cfg, "d2nn_paraxial", False))
        self.register_buffer("inputs", angle_scan_inputs(
            cfg.n_fine, cfg.pitch_fine, cfg.wavelength_um, cfg.n_alphas, cfg.n_thetas, device=dev))
        self.register_buffer("scan_dirs", angle_scan_directions(cfg.n_alphas, cfg.n_thetas, device=dev))
        self.exc = VectorialExcitation(
            cfg.n_fine, cfg.pitch_fine, cfg.dz_um, cfg.nz, cfg.na_ex, n_index=cfg.n_index,
            lambda_um=cfg.wavelength_um, bin_factor=cfg.bin_factor, w=cfg.exc_window, device=dev)
        # Objective back-aperture: low-pass the D2NN output field to NA_ex (cutoff NA/lambda).
        self.explicit_pupil = getattr(cfg, "explicit_pupil", False)
        if self.explicit_pupil:
            f = torch.fft.fftfreq(cfg.n_fine, d=cfg.pitch_fine, device=dev)   # cyc/um
            kr = torch.sqrt(f.view(-1, 1) ** 2 + f.view(1, -1) ** 2)
            self.register_buffer("pupil", (kr <= cfg.na_ex / cfg.wavelength_um).to(torch.complex64))

    @property
    def phases(self):
        """List of learnable phase masks (one per learnable D2NN layer)."""
        return self.d2nn.phases

    @property
    def focus_idx(self) -> int:
        return self.exc.focus_idx

    @property
    def n_patterns(self) -> int:
        return self.inputs.shape[0]

    @property
    def n_planes(self) -> int:
        return self.exc.nz

    def _pupil_filter(self, field):
        """Band-limit the field to NA_ex (the objective back-aperture)."""
        return torch.fft.ifft2(torch.fft.fft2(field) * self.pupil)

    def plane_field(self, idx=None, pupil=None, dist_um=None):
        """Complex field after the last layer: (T, N_fine, N_fine)."""
        field = self.d2nn(self.inputs if idx is None else self.inputs[idx], out_dz_um=dist_um)
        use = self.explicit_pupil if pupil is None else (pupil and self.explicit_pupil)
        return self._pupil_filter(field) if use else field

    def forward(self, idx=None):
        return self.exc.spread(self.plane_field(idx))


def build_model(cfg) -> DiffractiveCoder:
    return DiffractiveCoder(cfg).to(cfg.device)
