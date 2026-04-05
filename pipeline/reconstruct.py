"""
Multi-slice plug-and-play reconstruction.

Least-squares data term plus three orthogonal DnCNN slice denoisers and an L1 term, combined with ADMM.
"""
from __future__ import annotations

import scico.numpy as snp
from scico import linop, loss, functional, metric
from scico.optimize.admm import ADMM, LinearSubproblemSolver


class DnCNN_YZ(functional.DnCNN):
    """DnCNN on the physical xy slices of a (z, x, y) volume."""
    def prox(self, x, lam=1.0, **kw):
        xs = snp.transpose(x, axes=(2, 1, 0))
        d = super().prox(xs, lam, **kw)
        return snp.transpose(d, axes=(2, 1, 0))


class DnCNN_ZX(functional.DnCNN):
    """DnCNN on the physical yz slices of a (z, x, y) volume."""
    def prox(self, x, lam=1.0, **kw):
        xs = snp.transpose(x, axes=(2, 0, 1))
        d = super().prox(xs, lam, **kw)
        return snp.transpose(d, axes=(1, 2, 0))


# lmda_xy, lmda_yz, lmda_zx weight the default (zx), DnCNN_YZ (xy) and DnCNN_ZX (yz) denoisers
RECON = dict(variant="17M", rho=5.0, lmda=0.2, l1_rho=10.0, l1_lmda=0.2,
             lmda_xy=None, lmda_yz=None, lmda_zx=None,
             max_iter=20, f_iter=20, f_tol=1e-5)

# Named prior presets: overrides passed to run_pnp. "dncnn" is the default (3 DnCNN + L1); "l1"
# drops the denoisers for a sparsity-only prior. Shared by the comparison scripts.
PRIOR_PRESETS = {
    "dncnn": {},
    "l1":    {"lmda_xy": 0.0, "lmda_yz": 0.0, "lmda_zx": 0.0, "l1_lmda": 0.05},
}


def run_pnp(phi, measure, obj_shape, gt=None, n_measure=None, verbose=False, return_hist=False,
            phi_scale=1.0, callback=None, **over):
    """phi: (M, prod(obj_shape)) jax matrix."""
    cfg = {**RECON, **over}
    n_measure = phi.shape[0] if n_measure is None else n_measure
    if n_measure < phi.shape[0]:
        phi = phi[:n_measure, :]
    measure = measure[:n_measure]

    A = linop.LinearOperator(input_shape=obj_shape, output_shape=(n_measure,),
                             eval_fn=lambda x: phi_scale * (phi @ x.flatten()))
    f = loss.SquaredL2Loss(y=measure, A=A)

    v = cfg["variant"]; lm = cfg["lmda"]
    pick = lambda k: lm if cfg.get(k) is None else cfg[k]           # per-axis weight, None -> lmda
    terms = [(pick("lmda_xy"), functional.DnCNN(variant=v), cfg["rho"]),
             (pick("lmda_yz"), DnCNN_YZ(variant=v), cfg["rho"]),
             (pick("lmda_zx"), DnCNN_ZX(variant=v), cfg["rho"]),
             (cfg["l1_lmda"], functional.L1Norm(), cfg["l1_rho"])]
    terms = [(w, g, r) for (w, g, r) in terms if w > 0]             # drop zero-weight terms
    g_list = [w * g for (w, g, _) in terms]
    C_list = [linop.Identity(obj_shape)] * len(terms)
    rho_list = [r for (_, _, r) in terms]

    subsolver = LinearSubproblemSolver(cg_kwargs={"tol": cfg["f_tol"], "maxiter": cfg["f_iter"]})
    x0 = A.T @ measure
    x0 = x0 / (x0.max() + 1e-12)

    solver = ADMM(f=f, g_list=g_list, C_list=C_list, rho_list=rho_list, x0=x0,
                  maxiter=cfg["max_iter"], subproblem_solver=subsolver,
                  itstat_options={"display": verbose, "period": 1 if return_hist else 5})
    est = snp.clip(solver.solve(callback=callback), 0, 1)
    psnr = float(metric.psnr(gt, est)) if gt is not None else None
    if return_hist:
        try:
            hist = solver.itstat_object.history(transpose=True)
        except Exception:
            hist = None
        return est, psnr, hist
    return est, psnr
