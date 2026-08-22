"""
Learned denoisers for the PnP solver.

  DnCNN2D  residual 2D DnCNN applied to xy / xz / yz slices
  DnCNN3D  residual 3D DnCNN on whole reconstruction volumes
  UNet3D   small 3D U-Net alternative
"""
from __future__ import annotations
import torch
import torch.nn as nn


class _DnCNN(nn.Module):
    def __init__(self, conv, bn, depth, width, in_ch=1):
        super().__init__()
        self.in_ch = in_ch                    # 2 = non-blind: (noisy, sigma map), as FFDNet / DRUNet
        layers = [conv(in_ch, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [conv(width, width, 3, padding=1, bias=False), bn(width), nn.ReLU(inplace=True)]
        layers += [conv(width, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigma=None):         # predicts the noise; returns the denoised input
        if self.in_ch == 2:                   # sigma: scalar or per-sample tensor
            s = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
            s = s.reshape(-1, *[1] * (x.dim() - 1)) if s.dim() > 0 else s
            return x - self.net(torch.cat([x, s.expand_as(x)], 1))
        return x - self.net(x)


class DnCNN2D(_DnCNN):
    def __init__(self, depth=17, width=64, in_ch=1):
        super().__init__(nn.Conv2d, nn.BatchNorm2d, depth, width, in_ch)


class DnCNN3D(_DnCNN):
    def __init__(self, depth=12, width=48, in_ch=1):
        super().__init__(nn.Conv3d, nn.BatchNorm3d, depth, width, in_ch)


class UNet3D(nn.Module):
    def __init__(self, width=32, in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        def blk(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
                                 nn.Conv3d(o, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
        w = width
        self.e1, self.e2, self.e3 = blk(in_ch, w), blk(w, 2 * w), blk(2 * w, 4 * w)
        self.u2 = nn.ConvTranspose3d(4 * w, 2 * w, 2, stride=2); self.d2 = blk(4 * w, 2 * w)
        self.u1 = nn.ConvTranspose3d(2 * w, w, 2, stride=2); self.d1 = blk(2 * w, w)
        self.out = nn.Conv3d(w, 1, 1); self.pool = nn.MaxPool3d(2)

    def forward(self, x, sigma=None):
        inp = x
        if self.in_ch == 2:
            s = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
            s = s.reshape(-1, *[1] * (x.dim() - 1)) if s.dim() > 0 else s
            inp = torch.cat([x, s.expand_as(x)], 1)
        e1 = self.e1(inp); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return x - self.out(d1)


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    if ck.get("arch") == "unet3d":
        m = UNet3D(width=ck["width"], in_ch=ck.get("in_ch", 1))
        m.load_state_dict(ck["state"]); m.to(device).eval()
        return m, ck
    m = (DnCNN2D if ck["dim"] == 2 else DnCNN3D)(depth=ck["depth"], width=ck["width"], in_ch=ck.get("in_ch", 1))
    m.load_state_dict(ck["state"]); m.to(device).eval()
    return m, ck


@torch.no_grad()
def apply_volume(model, v, mode, sigma=None):
    """v: torch (z, x, y) on the model's device -> denoised volume, same shape."""
    f = (lambda t: model(t, sigma)) if getattr(model, "in_ch", 1) == 2 else model
    if mode == "3d":
        return f(v[None, None])[0, 0]
    if mode == "xy":                            # batch over z
        return f(v[:, None])[:, 0]
    if mode == "xz":                            # batch over y: (y, z, x)
        return f(v.permute(2, 0, 1)[:, None])[:, 0].permute(1, 2, 0)
    if mode == "yz":                            # batch over x: (x, z, y)
        return f(v.permute(1, 0, 2)[:, None])[:, 0].permute(1, 0, 2)
    raise ValueError(mode)


_CACHE = {}


def make_functional(path, mode, device="cuda"):
    """scico Functional applying the checkpoint at `path` in orientation `mode` (models cached per path)."""
    import jax
    import numpy as np
    from scico import functional

    if path not in _CACHE:
        _CACHE[path] = load_model(path, device)[0]
    model = _CACHE[path]

    class TorchDenoiser(functional.Functional):
        has_eval = False
        has_prox = True

        def prox(self, x, lam=1.0, **kwargs):
            def f(xn):
                t = torch.from_numpy(np.asarray(xn, dtype=np.float32)).to(device)
                return apply_volume(model, t, mode).cpu().numpy().astype(np.float32)
            return jax.pure_callback(f, jax.ShapeDtypeStruct(x.shape, np.float32), x).astype(x.dtype)

    return TorchDenoiser()
