"""
Reconstruction and debugging figures: numpy arrays in, matplotlib figures out.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


# --- helpers ---------------------------------------------------------------------------
def emit(fig, save=None):
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight"); plt.close(fig); print(f"figure -> {save}")
        return
    import sys
    if "ipykernel" in sys.modules:
        from IPython.display import display
        display(fig)
    else:
        plt.show()
    plt.close(fig)


def mip(vol):
    """Max-intensity projections: xy (over depth), xz (over y), yz (over x)."""
    return {"xy": vol.max(0), "xz": vol.max(2), "yz": vol.max(1)}


def _vmax(vols):
    return max(float(np.max(v)) for v in vols.values()) or 1.0


def _imshow(ax, img, vmax, cmap, aspect="equal"):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, aspect=aspect)
    ax.set_xticks([]); ax.set_yticks([])


# --- views -----------------------------------------------------------------------------
def show_mips(vols, cmap="inferno", title=None):
    vmax = _vmax(vols); names = list(vols)
    fig, ax = plt.subplots(len(names), 3, figsize=(9, 3 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        m = mip(vols[name])
        for c, key in enumerate(("xy", "xz", "yz")):
            _imshow(ax[r, c], m[key], vmax, cmap, aspect="auto" if key != "xy" else "equal")
            if r == 0:
                ax[r, c].set_title({"xy": "XY (over z)", "xz": "XZ (depth↑)",
                                    "yz": "YZ (depth↑)"}[key], fontsize=9)
        ax[r, 0].set_ylabel(name, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def show_slices(vols, cmap="inferno", title=None):
    """Row per volume, one column per depth plane (z-slice montage)."""
    vmax = _vmax(vols); names = list(vols); rz = next(iter(vols.values())).shape[0]
    fig, ax = plt.subplots(len(names), rz, figsize=(1.5 * rz, 1.6 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        for k in range(rz):
            _imshow(ax[r, k], vols[name][k], vmax, cmap)
            if r == 0:
                ax[r, k].set_title(f"z{k}", fontsize=8)
        ax[r, 0].set_ylabel(name, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def show_error(gt, est, name="est", cmap="inferno", cmap_err="magma"):
    """|est-GT| projections + a voxel scatter (est vs GT) with R^2 and NMSE."""
    err = np.abs(est - gt); me, mg = mip(err), mip(gt)
    nmse = float(np.sum((est - gt) ** 2) / (np.sum(gt ** 2) + 1e-12))
    g, e = gt.ravel(), est.ravel()
    r2 = float(np.corrcoef(g, e)[0, 1] ** 2) if g.std() > 0 and e.std() > 0 else 0.0
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    _imshow(ax[0], mg["xz"], float(gt.max()) or 1.0, cmap, aspect="auto"); ax[0].set_title("GT XZ", fontsize=9)
    for a, key in ((ax[1], "xz"), (ax[2], "yz")):
        _imshow(a, me[key], float(err.max()) or 1.0, cmap_err, aspect="auto")
        a.set_title(f"|{name}-GT| {key.upper()}", fontsize=9)
    ax[3].scatter(g, e, s=3, alpha=0.3); lim = max(g.max(), e.max()) or 1.0
    ax[3].plot([0, lim], [0, lim], "r--", lw=1)
    ax[3].set_xlabel("GT"); ax[3].set_ylabel(name); ax[3].set_aspect("equal")
    ax[3].set_title(f"R²={r2:.3f}  NMSE={nmse:.3f}", fontsize=9)
    fig.tight_layout()
    return fig


def show_depth_corr(A, obj_shape, title="operator depth-column correlation"):
    """rz x rz correlation between per-depth mean measurement responses of the forward operator A."""
    rz, rx, ry = obj_shape; per = rx * ry
    sig = np.stack([A[:, z * per:(z + 1) * per].mean(1) for z in range(rz)], 0)  # (rz, M)
    sig = sig - sig.mean(1, keepdims=True)
    sig = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-12)
    C = np.abs(sig @ sig.T)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(C, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("depth plane"); ax.set_ylabel("depth plane"); ax.set_title(title, fontsize=9)
    anti = np.mean([C[z, rz - 1 - z] for z in range(rz)])
    ax.text(0.5, -0.18, f"mean anti-diagonal |corr| = {anti:.2f}  (1 = full +d/-d degeneracy)",
            transform=ax.transAxes, ha="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def show_measurements(Y, meas_shape, ncols=8, cmap="gray", title="measurements Y"):
    """Montage of the T detector frames (the raw data the reconstruction sees)."""
    Y = np.asarray(Y).reshape(meas_shape); T = meas_shape[0]
    nrows = int(np.ceil(T / ncols)); vmax = float(Y.max()) or 1.0
    fig, ax = plt.subplots(nrows, ncols, figsize=(1.1 * ncols, 1.1 * nrows), squeeze=False)
    for i in range(nrows * ncols):
        a = ax[i // ncols, i % ncols]; a.axis("off")
        if i < T:
            a.imshow(Y[i], cmap=cmap, vmin=0, vmax=vmax)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    return fig


def show_convergence(hist, title="ADMM convergence"):
    """Objective and primal/dual residuals vs iteration (from run_pnp(..., return_hist=True))."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    it = getattr(hist, "Iter", None)
    x = it if it is not None else range(len(getattr(hist, "Objective", [])))
    if hasattr(hist, "Objective"):
        ax[0].plot(x, hist.Objective, "o-"); ax[0].set_ylabel("objective")
    for field, lab in (("Prml_Rsdl", "primal"), ("Dual_Rsdl", "dual")):
        if hasattr(hist, field):
            ax[1].semilogy(x, getattr(hist, field), "o-", label=lab)
    ax[1].set_ylabel("residual (log)"); ax[1].legend()
    for a in ax:
        a.set_xlabel("iteration"); a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    return fig


def overlay_axial_profile(gt_profile, est_profiles, title=None):
    """GT vs each estimate's normalised per-plane energy (est_profiles: {name: (rz,)})."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(gt_profile, "k-o", lw=2, label="GT")
    for name, p in est_profiles.items():
        ax.plot(p, "o--", label=name)
    ax.set_xlabel("depth plane"); ax.set_ylabel("normalised energy"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def dashboard(gt, est, name="est", axial=None, title=None):
    vmax = max(float(gt.max()), float(est.max())) or 1.0
    mg, me = mip(gt), mip(est)
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 4)
    for c, key in enumerate(("xy", "xz", "yz")):
        a = fig.add_subplot(gs[0, c]); _imshow(a, mg[key], vmax, "inferno",
                                                aspect="auto" if key != "xy" else "equal")
        a.set_title(f"GT {key.upper()}", fontsize=9)
        b = fig.add_subplot(gs[1, c]); _imshow(b, me[key], vmax, "inferno",
                                                aspect="auto" if key != "xy" else "equal")
        b.set_title(f"{name} {key.upper()}", fontsize=9)
    if axial is not None:
        a = fig.add_subplot(gs[0, 3]); a.plot(axial[0], "k-o", lw=2); a.plot(axial[1], "o--")
        a.set_title("axial profile", fontsize=9); a.grid(alpha=0.3); a.set_xlabel("depth")
    ax = fig.add_subplot(gs[1, 3]); g, e = gt.ravel(), est.ravel()
    ax.scatter(g, e, s=3, alpha=0.3); lim = max(g.max(), e.max()) or 1.0
    ax.plot([0, lim], [0, lim], "r--", lw=1); ax.set_aspect("equal")
    r2 = float(np.corrcoef(g, e)[0, 1] ** 2) if g.std() > 0 and e.std() > 0 else 0.0
    ax.set_title(f"voxels R²={r2:.3f}", fontsize=9); ax.set_xlabel("GT"); ax.set_ylabel(name)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
