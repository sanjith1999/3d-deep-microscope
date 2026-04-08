"""
Depth metrics on (z, x, y) volumes: axial profile, centroid, local depth error, mirror-ghost energy.
"""
from __future__ import annotations
import numpy as np
import scico
import scico.numpy as snp


def axial_profile(vol):
    """Normalised per-plane energy: (rz,) summing to 1."""
    p = vol.reshape(vol.shape[0], -1).sum(1)
    return p / (p.sum() + 1e-12)


def axial_centroid(vol):
    """Energy-weighted depth-plane centroid (in plane units)."""
    p = axial_profile(vol); z = np.arange(len(p))
    return float((z * p).sum())


def depth_content(vol):
    p = axial_profile(vol); z = np.arange(len(p)); c = (z * p).sum()
    spread = float(np.sqrt(((z - c) ** 2 * p).sum()))
    P = vol.reshape(vol.shape[0], -1); P = P - P.mean(1, keepdims=True)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    G = np.abs(P @ P.T); off = G[~np.eye(len(p), dtype=bool)]
    return {"axial_spread": spread, "plane_decorrelation": float(1 - off.mean())}


def depth_metrics(gt, est):
    """PSNR3D plus depth-specific error: axial-centroid error and per-plane energy-profile L1."""
    return {"psnr": float(scico.metric.psnr(snp.array(gt), snp.array(est))),
            "centroid_err": abs(axial_centroid(gt) - axial_centroid(est)),
            "profile_l1": float(np.abs(axial_profile(gt) - axial_profile(est)).sum())}


def local_depth_err(gt, est, frac=0.1):
    """Energy-weighted mean |centroid_gt - centroid_est| (planes) over lateral columns carrying signal."""
    z = np.arange(gt.shape[0])[:, None, None]
    g, e = gt.sum(0), est.sum(0)
    mask = g > frac * g.max()
    cg = (z * gt).sum(0) / (g + 1e-12)
    ce = (z * est).sum(0) / (e + 1e-12)
    w = g[mask]
    return float((np.abs(cg - ce)[mask] * w).sum() / (w.sum() + 1e-12))


def depth_blind_oracle(gt):
    """Exact lateral structure + exact global axial profile, zero local depth information."""
    return gt.sum(0, keepdims=True) * axial_profile(gt)[:, None, None]


def score(gt, est):
    """depth_metrics + axial-shape ratio + local depth error."""
    m = depth_metrics(gt, est)
    m["shape"] = float(axial_profile(est).std() / (axial_profile(gt).std() + 1e-12))
    m["local_depth_err"] = local_depth_err(gt, est)
    return m


def bead_report(est, beads, mirror=lambda rz: 14 - rz):
    rows = []
    for (rz, rx, ry) in beads:
        col = est[:, max(rx - 1, 0):rx + 2, max(ry - 1, 0):ry + 2].sum((1, 2))
        mz = mirror(rz)
        win = lambda k: col[max(k - 1, 0):k + 2].sum()
        rows.append({"bead": [rz, rx, ry], "peak_z": int(col.argmax()), "true_z": rz, "mirror_z": mz,
                     "ghost_ratio": float(win(mz) / (win(rz) + 1e-12))})
    return rows
