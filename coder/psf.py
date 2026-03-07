"""
Vectorial Richards-Wolf PSF of the high-NA excitation objective and FFT-based 3D convolution.
"""
from __future__ import annotations
import numpy as np
import scipy.special
import torch
from torch import nn


def calculate_phi(NPXLS: int) -> torch.Tensor:
    """Azimuthal angle map phi(x, y) about the grid centre, for the vectorial PSF."""
    N = (NPXLS - 1) / 2
    A = torch.arange(-N, N + 1)
    XX = torch.unsqueeze(A, dim=0)
    YY = -torch.unsqueeze(A, dim=1)
    X = torch.tile(XX, (NPXLS, 1))
    Y = torch.tile(YY, (1, NPXLS))
    return torch.atan2(Y, X)


def conv_3D(PSF_3D: torch.Tensor, H: torch.Tensor, w: int = 1) -> torch.Tensor:
    """Linear 3D convolution via zero-padded FFT."""
    Bp, nz, nx, ny = PSF_3D.shape
    Bh, _, _, _ = H.shape
    PSF_3D_PAD = torch.zeros(Bp, w * nz, w * nx, w * ny).type(torch.complex64).to(PSF_3D.device)
    H_PAD = torch.zeros(Bh, w * nz, w * nx, w * ny).type(torch.complex64).to(H.device)

    PSF_3D_PAD[:, (w - 1) * nz // 2:nz + (w - 1) * nz // 2, (w - 1) * nx // 2:nx + (w - 1) * nx // 2, (w - 1) * ny // 2:ny + (w - 1) * ny // 2] = PSF_3D
    H_PAD[:, (w - 1) * nz // 2:nz + (w - 1) * nz // 2, (w - 1) * nx // 2:nx + (w - 1) * nx // 2, (w - 1) * ny // 2:ny + (w - 1) * ny // 2] = H

    Ht_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(H_PAD, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    PSF_fft = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(PSF_3D_PAD, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    conv_PSF_H_PAD = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(PSF_fft * Ht_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1)), dim=(-3, -2, -1))

    conv_PSF_H = conv_PSF_H_PAD[:, (w - 1) * nz // 2:(w - 1) * nz // 2 + nz, (w - 1) * nx // 2:(w - 1) * nx // 2 + nx, (w - 1) * ny // 2:(w - 1) * ny // 2 + ny]
    return conv_PSF_H


class psf_model(nn.Module):
    """Vectorial Richards-Wolf PSF for an x-polarised input; forward() -> (3, Nx, Ny, Nz)."""

    def __init__(self, NA, Rindex, lambda_, dx, dy, dz, Nx, Ny, Nz):
        super().__init__()
        self.NA = NA
        self.Rindex = Rindex
        self.lambda_ = lambda_
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Ntheta = 400  # n grid in polar angle

        self.abberations = torch.nn.Parameter(torch.zeros((self.Ntheta,)).float())
        self.init_psf_params()

    def init_psf_params(self):
        self.alpha = np.arcsin(self.NA / self.Rindex)

        self.x = self.dx * torch.arange(-self.Nx // 2 + 1, self.Nx // 2 + 1)
        self.y = self.dy * torch.arange(-self.Ny // 2 + 1, self.Ny // 2 + 1)
        self.z = self.dz * torch.arange(-self.Nz // 2 + 1, self.Nz // 2 + 1)

        self.dtheta = self.alpha / self.Ntheta
        self.theta = torch.arange(0, self.Ntheta) * self.dtheta

        assert self.Nx == self.Ny, "self.Nx != self.Ny -> self.Phi calculation wrong !!!"
        self.Phi = calculate_phi(self.Nx)
        self.A = np.pi / self.lambda_

        self.X, self.Y, self.THETA = torch.meshgrid(self.x, self.y, self.theta, indexing='ij')

        V = (2 * np.pi / self.lambda_) * torch.sqrt(self.X ** 2 + self.Y ** 2).numpy()  # k.r
        J0 = torch.from_numpy(scipy.special.jv(0, V * np.sin(self.THETA.numpy())))
        J1 = torch.from_numpy(scipy.special.jv(1, V * np.sin(self.THETA.numpy())))
        J2 = torch.from_numpy(scipy.special.jv(2, V * np.sin(self.THETA.numpy())))

        self.Func0 = torch.sqrt(torch.cos(self.THETA)) * torch.sin(self.THETA) * (1 + torch.cos(self.THETA)) * J0
        self.Func1 = torch.sqrt(torch.cos(self.THETA)) * (torch.sin(self.THETA) ** 2) * J1
        self.Func2 = torch.sqrt(torch.cos(self.THETA)) * torch.sin(self.THETA) * (1 - torch.cos(self.THETA)) * J2

        self.U = 2 * np.pi / self.lambda_ * self.z  # k.z

    def forward(self):
        device = self.abberations.device
        ABBR = torch.tile(self.abberations, (self.x.shape[0], self.y.shape[0], 1))

        Func0 = self.Func0.to(device) * torch.exp(1j * ABBR)
        Func1 = self.Func1.to(device) * torch.exp(1j * ABBR)
        Func2 = self.Func2.to(device) * torch.exp(1j * ABBR)

        PSF_3D = torch.zeros((3, self.Nx, self.Ny, self.Nz), dtype=torch.cfloat).to(device)

        for k in range(len(self.U)):
            Func3_atThisU = torch.exp(-1j * self.U[k] * torch.cos(self.THETA)).to(device)

            I0 = torch.trapz(y=Func0 * Func3_atThisU, x=self.theta.to(device), axis=2)  # axis: alpha
            I1 = torch.trapz(y=Func1 * Func3_atThisU, x=self.theta.to(device), axis=2)  # axis: alpha
            I2 = torch.trapz(y=Func2 * Func3_atThisU, x=self.theta.to(device), axis=2)  # axis: alpha

            Ex = 1j * self.A * (I0 + I2 * torch.cos(2 * self.Phi.to(device)))
            Ey = 1j * self.A * I2 * torch.sin(2 * self.Phi.to(device))
            Ez = (-2) * self.A * I1 * torch.cos(self.Phi.to(device))

            PSF_3D[0, :, :, k] = Ex
            PSF_3D[1, :, :, k] = Ey
            PSF_3D[2, :, :, k] = Ez

        return PSF_3D
