"""
Interactive forward-model lab: object-plane fields, 3D illumination, mirror planes and operator columns per coder.
"""
from __future__ import annotations
import math, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from configs.default import get_config
from coder import baselines
from coder.diffractive import _angle_grid
from coder.excitation import VectorialExcitation
from pipeline.forward import ExportForward, emission_psf, matched_dose
from pipeline.specimens import mirror_plane
from pipeline.viz import emit


class Coder:
    """One illumination coder: the 4F object-plane fields of its angle scan and their 3D relay."""

    def __init__(self, lab, kind, seed=0, ckpt=None, label=None):
        self.lab, self.kind, self.seed, self.ckpt = lab, kind, seed, ckpt
        self.label = label or (kind if kind != "designed" else Path(ckpt).parent.parent.name)
        cfg = lab.cfg
        if kind == "dmd":
            self.model, self._dmd = None, baselines.dmd_fields(cfg, seed)
        elif kind in ("random", "designed"):
            self.model = baselines.d2nn_model(cfg, seed=seed, coder_path=ckpt if kind == "designed" else None)
            self.model.requires_grad_(False)
            self.model.exc = lab.exc          # share one relay; drop the per-model copy
        else:
            raise ValueError(f"kind must be dmd | random | designed, got {kind!r}")
        self._H2 = None

    def __repr__(self):
        return f"Coder({self.label})"

    @property
    def n_layers(self):
        return 0 if self.model is None else len(self.model.phases)

    @torch.no_grad()
    def raw_field(self, idx):
        """Object-plane field BEFORE the NA-0.8 relay cut (N_fine, N_fine) complex."""
        if self.model is None:
            return self._dmd[idx]
        i = torch.tensor([idx], device=self.lab.dev)
        return self.model.plane_field(i, pupil=False)[0]

    @torch.no_grad()
    def field(self, idx):
        """Object-plane field AFTER the NA-0.8 relay cut: what the 4F relay images into the sample."""
        return self.lab.pupil_filter(self.raw_field(idx))

    @torch.no_grad()
    def volume(self, idx):
        """3D illumination |E|^2 of one pattern on the analysis grid (nz, 96, 96), unit input."""
        f = self.raw_field(idx) if self.model is None else self.field(idx)   # as in the build (baselines)
        return self.lab.exc.spread(f[None])[0]

    @torch.no_grad()
    def H2(self):
        """All patterns (T, nz, 96, 96), dose-normalised exactly like the operator build."""
        if self._H2 is None:
            T, dev, parts = self.lab.T, self.lab.dev, []
            for s in range(0, T, 16):
                i = torch.arange(s, min(s + 16, T), device=dev)
                f = self._dmd[i] if self.model is None else self.model.plane_field(i, pupil=True)
                parts.append(self.lab.exc.spread(f))
            self._H2 = matched_dose(torch.cat(parts), self.lab.cfg.dose_mean)
            if dev == "cuda":
                torch.cuda.empty_cache()
        return self._H2

    def light_kept(self):
        """Mean relay-plane intensity over the scan (the equal-laser-power scale)."""
        return baselines.relay_light(self.lab.cfg, self.kind, seed=self.seed, coder_path=self.ckpt)


class Lab:
    def __init__(self, device=None, cfg=None):
        self.cfg = cfg or get_config()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg.device = self.dev
        c = self.cfg
        self.T = c.n_alphas * c.n_thetas
        self.Nc = c.n_fine // c.bin_factor
        self.exc = VectorialExcitation(c.n_fine, c.pitch_fine, c.dz_um, c.nz, c.na_ex, n_index=c.n_index,
                                       lambda_um=c.wavelength_um, bin_factor=c.bin_factor, w=c.exc_window,
                                       device=self.dev, checkpoint_planes=False)
        f = torch.fft.fftfreq(c.n_fine, d=c.pitch_fine, device=self.dev)
        self.kr = torch.sqrt(f.view(-1, 1) ** 2 + f.view(1, -1) ** 2)
        self.pupil = (self.kr <= c.na_ex / c.wavelength_um).to(torch.complex64)
        self.focus = self.exc.focus_idx
        self.z_um = (np.arange(c.nz) - self.focus) * c.dz_um        # depth of each illumination plane
        alphas, thetas = _angle_grid(c.n_alphas, c.n_thetas)
        self.angles = [(math.degrees(a), math.degrees(t)) for a in alphas for t in thetas]
        self._emPSF = None
        print(f"[fwdlab] device={self.dev} patterns={self.T} fine {c.n_fine}@{c.pitch_fine}um -> "
              f"analysis {self.Nc}@{c.pitch_scalar}um, nz={c.nz}@{c.dz_um}um (focus plane {self.focus}), "
              f"gap {c.d2nn_out_dist_um}um, NA {c.na_ex}")

    def coder(self, kind, seed=0, ckpt=None, label=None) -> Coder:
        return Coder(self, kind, seed=seed, ckpt=None if ckpt is None else str(_ROOT / ckpt), label=label)

    def pupil_filter(self, field):
        return torch.fft.ifft2(torch.fft.fft2(field) * self.pupil)

    def angle(self, idx):
        a, t = self.angles[idx]
        return f"#{idx}  alpha={a:.1f}deg theta={t:.1f}deg"

    def plane(self, d_um):
        """Illumination plane index nearest depth d_um (0 = focal plane)."""
        return int(np.clip(self.focus + round(d_um / self.cfg.dz_um), 0, self.cfg.nz - 1))

    # --- views --------------------------------------------------------------------------
    def show_scan(self, idx=None, save=None):
        """The angle scan in sine space (cos(theta)*(cos a, sin a)); idx highlighted."""
        from coder.diffractive import angle_scan_directions
        d = angle_scan_directions(self.cfg.n_alphas, self.cfg.n_thetas).numpy()
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.scatter(d[:, 0], d[:, 1], s=10, c=np.arange(len(d)), cmap="viridis")
        if idx is not None:
            ax.scatter(*d[idx], s=120, facecolors="none", edgecolors="r", lw=2)
            ax.set_title(self.angle(idx), fontsize=9)
        ax.set_aspect("equal"); ax.set_xlabel("sx"); ax.set_ylabel("sy"); ax.grid(alpha=0.3)
        fig.tight_layout(); emit(fig, save)

    def show_layers(self, coder, save=None):
        """The D2NN phase masks (0.32 um neurons)."""
        if coder.model is None:
            print("DMD has no phase layers"); return
        ph = [p.detach().cpu().numpy() % (2 * np.pi) for p in coder.model.phases]
        fig, ax = plt.subplots(1, len(ph), figsize=(3.4 * len(ph), 3.2), squeeze=False)
        for i, p in enumerate(ph):
            im = ax[0, i].imshow(p, cmap="twilight", vmin=0, vmax=2 * np.pi)
            ax[0, i].set_title(f"{coder.label} layer {i + 1}", fontsize=9); ax[0, i].axis("off")
        fig.colorbar(im, ax=ax[0, -1], fraction=0.046); fig.tight_layout(); emit(fig, save)

    def show_plane(self, coders, idx=0, save=None):
        coders = [coders] if isinstance(coders, Coder) else coders
        c = self.cfg; ext = c.n_fine * c.pitch_fine
        fig, ax = plt.subplots(len(coders), 4, figsize=(14, 3.4 * len(coders)), squeeze=False)
        fmax = 1 / (2 * c.pitch_fine); na_f = c.na_ex / c.wavelength_um
        for r, cd in enumerate(coders):
            raw, cut = cd.raw_field(idx), cd.field(idx)
            I0, I1 = raw.abs().square().cpu().numpy(), cut.abs().square().cpu().numpy()
            spec = torch.fft.fftshift(torch.fft.fft2(raw)).abs().square().cpu().numpy()
            vmax = float(np.percentile(I1, 99.5)) or 1.0
            ax[r, 0].imshow(I0, cmap="inferno", vmin=0, vmax=max(vmax, float(np.percentile(I0, 99.5))),
                            extent=[0, ext, ext, 0])
            ax[r, 0].set_title(f"{cd.label}: |E|^2 before cut (mean {I0.mean():.3f})", fontsize=9)
            ax[r, 1].imshow(I1, cmap="inferno", vmin=0, vmax=vmax, extent=[0, ext, ext, 0])
            ax[r, 1].set_title(f"after NA cut (light kept {I1.mean():.3f})", fontsize=9)
            ax[r, 2].imshow(np.angle(cut.cpu().numpy()), cmap="twilight", extent=[0, ext, ext, 0])
            ax[r, 2].set_title("phase after cut", fontsize=9)
            ax[r, 3].imshow(np.log10(spec + spec.max() * 1e-8), cmap="magma", extent=[-fmax, fmax, fmax, -fmax])
            ax[r, 3].add_patch(plt.Circle((0, 0), na_f, fill=False, color="c", lw=1))
            ax[r, 3].set_title("angular spectrum (log), NA 0.8 circle", fontsize=9)
            ax[r, 0].set_ylabel("um")
        fig.suptitle(f"4F object plane ({c.d2nn_out_dist_um} um after last layer for D2NN)   {self.angle(idx)}",
                     fontsize=10)
        fig.tight_layout(); emit(fig, save)

    def show_volume(self, coders, idx=0, depths_um=(-3.84, -1.92, 0.0, 1.92, 3.84), y=None, save=None):
        """3D illumination of one pattern: XY at chosen depths + XZ cut (at row y) and YZ cut."""
        coders = [coders] if isinstance(coders, Coder) else coders
        c = self.cfg; ext = self.Nc * c.pitch_scalar; y = self.Nc // 2 if y is None else y
        zext = [0, ext, self.z_um[0], self.z_um[-1]]      # data flipped so +z is at the top
        n = len(depths_um)
        fig, ax = plt.subplots(len(coders), n + 2, figsize=(2.3 * (n + 2), 2.6 * len(coders)), squeeze=False)
        for r, cd in enumerate(coders):
            V = cd.volume(idx).cpu().numpy(); vmax = float(np.percentile(V, 99.5)) or 1.0
            for k, d in enumerate(depths_um):
                p = self.plane(d)
                ax[r, k].imshow(V[p], cmap="inferno", vmin=0, vmax=vmax, extent=[0, ext, ext, 0])
                ax[r, k].set_title(f"z={self.z_um[p]:+.2f}um", fontsize=8); ax[r, k].axis("off")
            ax[r, n].imshow(V[:, y, :][::-1], cmap="inferno", vmin=0, vmax=vmax, extent=zext, aspect="auto")
            ax[r, n].set_title(f"XZ (row {y}), z up", fontsize=8)
            ax[r, n + 1].imshow(V[:, :, y][::-1], cmap="inferno", vmin=0, vmax=vmax, extent=zext, aspect="auto")
            ax[r, n + 1].set_title(f"YZ (col {y}), z up", fontsize=8)
            ax[r, 0].text(-0.1, 0.5, cd.label, transform=ax[r, 0].transAxes, rotation=90, va="center",
                          ha="right", fontsize=9)
        fig.suptitle(f"3D illumination |E|^2 (analysis grid)   {self.angle(idx)}", fontsize=10)
        fig.tight_layout(); emit(fig, save)

    def show_mirror(self, coders, idx=0, d_um=2.56, save=None):
        """One pattern at +d and -d about the focal plane, and their difference."""
        coders = [coders] if isinstance(coders, Coder) else coders
        pp, pm = self.plane(d_um), self.plane(-d_um)
        fig, ax = plt.subplots(len(coders), 3, figsize=(9.5, 3.2 * len(coders)), squeeze=False)
        for r, cd in enumerate(coders):
            V = cd.volume(idx).cpu().numpy(); a, b = V[pp], V[pm]
            vmax = float(np.percentile(V[[pp, pm]], 99.5)) or 1.0
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            ax[r, 0].imshow(a, cmap="inferno", vmin=0, vmax=vmax); ax[r, 0].set_title(f"{cd.label} z={self.z_um[pp]:+.2f}um", fontsize=9)
            ax[r, 1].imshow(b, cmap="inferno", vmin=0, vmax=vmax); ax[r, 1].set_title(f"z={self.z_um[pm]:+.2f}um", fontsize=9)
            lim = float(np.abs(a - b).max()) or 1.0
            ax[r, 2].imshow(a - b, cmap="RdBu_r", vmin=-lim, vmax=lim); ax[r, 2].set_title(f"difference   corr={corr:.3f}", fontsize=9)
            for x in ax[r]: x.axis("off")
        fig.suptitle(f"+d / -d mirror   {self.angle(idx)}", fontsize=10); fig.tight_layout(); emit(fig, save)

    def depth_corr(self, coder, n_pix=512, seed=0):
        H = coder.H2()
        g = torch.Generator().manual_seed(seed)
        pix = torch.randint(0, self.Nc * self.Nc, (n_pix,), generator=g)
        S = H.reshape(H.shape[0], H.shape[1], -1)[:, :, pix.to(H.device)].float()   # (T, nz, n_pix)
        S = S - S.mean(0, keepdim=True); S = S / (S.norm(dim=0, keepdim=True) + 1e-12)
        return torch.einsum("tap,tbp->abp", S, S).abs().mean(-1).cpu().numpy()

    def show_depth_corr(self, coders, n_pix=512, save=None):
        """Per-pixel depth correlation over the whole angle scan."""
        coders = [coders] if isinstance(coders, Coder) else coders
        fig, ax = plt.subplots(1, len(coders), figsize=(4.2 * len(coders), 3.8), squeeze=False)
        e = [self.z_um[0], self.z_um[-1], self.z_um[-1], self.z_um[0]]
        out = {}
        for k, cd in enumerate(coders):
            C = self.depth_corr(cd, n_pix)
            anti = np.mean([C[self.focus + j, self.focus - j] for j in range(1, min(self.focus, self.cfg.nz - 1 - self.focus) + 1)])
            im = ax[0, k].imshow(C, cmap="viridis", vmin=0, vmax=1, extent=e)
            ax[0, k].set_title(f"{cd.label}: mirror |corr| {anti:.2f}", fontsize=9); out[cd.label] = float(anti)
            ax[0, k].set_xlabel("depth um"); ax[0, k].set_ylabel("depth um")
        fig.colorbar(im, ax=ax[0, -1], fraction=0.046); fig.tight_layout(); emit(fig, save)
        return out

    # --- operator columns ---------------------------------------------------------------
    def emPSF(self):
        if self._emPSF is None:
            c = self.cfg
            self._emPSF = emission_psf(c.nz, self.Nc, c.dz_um, c.pitch_scalar, c.na_em, device=self.dev)
        return self._emPSF

    @torch.no_grad()
    def column(self, coder, z, x, y, d_size=8):
        """Column of A for recon voxel (z, x, y): its detector response to every pattern, (T, Nd, Nd)."""
        c = self.cfg; patch = tuple(c.recon_patch)
        fwd = ExportForward(coder.H2(), self.emPSF(), patch=patch, d_size=d_size, chunk=16,
                            dose_mean=None, device=self.dev)
        e = torch.zeros(fwd.recon_shape, device=self.dev); e[z, x, y] = 1.0
        return fwd.forward(e).cpu().numpy()

    def show_column(self, coders, z=3, x=24, y=24, d_size=8, n_show=8, save=None):
        coders = [coders] if isinstance(coders, Coder) else coders
        zm = mirror_plane(z); pz = self.cfg.recon_patch[0]
        dz = lambda r: (r * pz + (pz - 1) / 2 - self.focus) * self.cfg.dz_um
        fig = plt.figure(figsize=(1.3 * n_show + 5, 2.9 * len(coders)))
        gs = fig.add_gridspec(2 * len(coders), n_show + 4)
        rows = {}
        for r, cd in enumerate(coders):
            a, b = self.column(cd, z, x, y, d_size), self.column(cd, zm, x, y, d_size)
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]); rows[cd.label] = corr
            vmax = max(a[:n_show].max(), b[:n_show].max()) or 1.0
            for k in range(n_show):
                for j, col in enumerate((a, b)):
                    q = fig.add_subplot(gs[2 * r + j, k]); q.imshow(col[k], cmap="gray", vmin=0, vmax=vmax)
                    q.set_xticks([]); q.set_yticks([])
                    if k == 0:
                        q.set_ylabel(f"{cd.label}\nz={dz([z, zm][j]):+.2f}um", fontsize=7)
                    if r == 0 and j == 0:
                        q.set_title(f"p{k}", fontsize=7)
            q = fig.add_subplot(gs[2 * r:2 * r + 2, n_show:])
            q.plot(a.reshape(a.shape[0], -1).sum(1), lw=1, label=f"z={dz(z):+.2f}um")
            q.plot(b.reshape(b.shape[0], -1).sum(1), lw=1, label=f"mirror z={dz(zm):+.2f}um")
            q.set_title(f"{cd.label}: total signal per pattern, column corr {corr:.3f}", fontsize=8)
            q.set_xlabel("pattern"); q.legend(fontsize=7)
        fig.suptitle(f"operator columns for voxel (x={x}, y={y}) at recon plane {z} and its mirror {zm}", fontsize=10)
        fig.tight_layout(); emit(fig, save)
        return rows
