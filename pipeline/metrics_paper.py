"""
Structure-aware metrics: 3D SSIM, NRMSE, foreground SSIM, bead localisation and directional FSC.
"""
from __future__ import annotations
import numpy as np

PITCH = 0.64


def nrmse(gt, est):
    return float(np.linalg.norm(est - gt) / (np.linalg.norm(gt) + 1e-12))


def ssim3d(gt, est, data_range=1.0):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(gt.astype(np.float64), est.astype(np.float64), data_range=data_range,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


def ssim3d_foreground(gt, est, data_range=1.0, frac=0.1, dilate=2):
    from skimage.metrics import structural_similarity
    from scipy.ndimage import binary_dilation
    _, smap = structural_similarity(gt.astype(np.float64), est.astype(np.float64), data_range=data_range,
                                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True)
    mask = binary_dilation(gt > frac * gt.max(), iterations=dilate)
    return float(smap[mask].mean()) if mask.any() else float("nan")


def _local_max(v, frac):
    from scipy.ndimage import maximum_filter
    m = (v == maximum_filter(v, size=3, mode="nearest")) & (v > frac * v.max())
    return np.argwhere(m)


def _centroid(v, p):
    z, x, y = p; sl = tuple(slice(max(c - 1, 0), c + 2) for c in (z, x, y))
    w = v[sl]; g = np.meshgrid(*[np.arange(s.start, s.start + n) for s, n in zip(sl, w.shape)], indexing="ij")
    return np.array([(w * gi).sum() / (w.sum() + 1e-12) for gi in g])


def bead_localisation(est, beads, frac=0.3, tol_lat=2, tol_ax=2, gt=None, smooth=0.7):
    """beads: list of (z, x, y) on the reconstruction grid."""
    from scipy.ndimage import gaussian_filter
    det = _local_max(gaussian_filter(est, smooth) if smooth else est, frac); used = set(); errs = []
    refs = [(_centroid(gt, np.array(b)) if gt is not None else np.array(b, float)) for b in beads]
    for b, ref in zip(beads, refs):
        b = np.array(b)
        cand = [(i, d) for i, d in enumerate(det) if i not in used
                and abs(d[0] - b[0]) <= tol_ax and max(abs(d[1] - b[1]), abs(d[2] - b[2])) <= tol_lat]
        if not cand:
            continue
        i, d = min(cand, key=lambda t: np.sum((t[1] - b) ** 2)); used.add(i)
        c = _centroid(est, d)
        errs.append((c[0] - ref[0], np.hypot(c[1] - ref[1], c[2] - ref[2])))
    tp = len(errs); fp = len(det) - tp; fn = len(beads) - tp
    e = np.array(errs) if errs else np.zeros((0, 2))
    return {"jaccard": tp / max(tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn,
            "rms_axial_um": float(np.sqrt(np.mean(e[:, 0] ** 2)) * PITCH) if tp else float("nan"),
            "rms_lateral_um": float(np.sqrt(np.mean(e[:, 1] ** 2)) * PITCH) if tp else float("nan")}


def fsc_directional(a, b, cone_deg=35.0, threshold=1 / 7, nbins=None):
    """FSC between volumes a, b (z, x, y) in a lateral and an axial cone."""
    A = np.fft.fftn(a - a.mean()); B = np.fft.fftn(b - b.mean())
    fz, fx, fy = [np.fft.fftfreq(n, d=PITCH) for n in a.shape]
    FZ, FX, FY = np.meshgrid(fz, fx, fy, indexing="ij")
    r = np.sqrt(FZ ** 2 + FX ** 2 + FY ** 2); ang = np.degrees(np.arctan2(np.abs(FZ), np.hypot(FX, FY)))
    fmax = 0.5 / PITCH; nbins = nbins or min(a.shape) // 2
    edges = np.linspace(0, fmax, nbins + 1); fc = 0.5 * (edges[1:] + edges[:-1])
    out = {"freq_cyc_per_um": fc.tolist()}
    for name, sel in (("lateral", ang <= 90 - cone_deg), ("axial", ang >= cone_deg)):
        curve = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sel & (r >= lo) & (r < hi) & (r > 0)
            num = np.real((A[m] * np.conj(B[m])).sum())
            den = np.sqrt((np.abs(A[m]) ** 2).sum() * (np.abs(B[m]) ** 2).sum())
            curve.append(float(num / den) if den > 0 else float("nan"))
        c = np.array(curve); below = np.where(np.isfinite(c) & (c < threshold))[0]
        f_res = fc[below[0]] if below.size else fc[-1]
        out[f"fsc_{name}"] = curve
        out[f"resolution_{name}_um"] = float(1 / f_res) if f_res > 0 else float("inf")
    return out


def fidelity_directional(gt, est, cone_deg=35.0, threshold=0.5, nbins=None):
    """Per-shell spectral fidelity against the ground truth, lateral and axial cones (see module doc)."""
    G = np.fft.fftn(gt - gt.mean()); E = np.fft.fftn(est - est.mean())
    fz, fx, fy = [np.fft.fftfreq(n, d=PITCH) for n in gt.shape]
    FZ, FX, FY = np.meshgrid(fz, fx, fy, indexing="ij")
    r = np.sqrt(FZ ** 2 + FX ** 2 + FY ** 2); ang = np.degrees(np.arctan2(np.abs(FZ), np.hypot(FX, FY)))
    fmax = 0.5 / PITCH; nbins = nbins or min(gt.shape) // 2
    edges = np.linspace(0, fmax, nbins + 1); fc = 0.5 * (edges[1:] + edges[:-1])
    out = {"freq_cyc_per_um": fc.tolist()}
    for name, sel in (("lateral", ang <= 90 - cone_deg), ("axial", ang >= cone_deg)):
        curve = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sel & (r >= lo) & (r < hi) & (r > 0)
            den = (np.abs(G[m]) ** 2).sum()
            curve.append(float(1 - (np.abs(E[m] - G[m]) ** 2).sum() / den) if den > 0 else float("nan"))
        c = np.array(curve); below = np.where(np.isfinite(c) & (c < threshold))[0]
        f_res = fc[below[0]] if below.size else fc[-1]
        out[f"fidelity_{name}"] = curve
        out[f"resolution_{name}_um"] = float(1 / f_res) if f_res > 0 else float("inf")
    return out
