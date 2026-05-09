"""
Interactive reconstruction lab over registered operators (results/recon/ops/).
"""
from __future__ import annotations
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json, math, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import scico.numpy as snp

from pipeline.reconstruct import run_pnp, PRIOR_PRESETS
from pipeline.metrics import axial_profile, score, depth_blind_oracle, bead_report
from pipeline.specimens import BEADS, mirror_plane
from pipeline import viz
from pipeline.viz import emit

PRESETS = {**PRIOR_PRESETS,
           "no_z":  {"lmda_yz": 0.0, "lmda_zx": 0.0},
           "xy_l1": {"lmda_yz": 0.0, "lmda_zx": 0.0, "l1_lmda": 0.05}}
# Wiener-metric sample columns: 4x4 lateral positions x every depth (as in the metric validation)
LAT = [(x, y) for x in (6, 18, 30, 42) for y in (6, 18, 30, 42)]


class Lab:
    def __init__(self, ops_dir="results/recon/ops"):
        self.ops_dir = _ROOT / ops_dir
        self.meta = {p.parent.name: json.loads(p.read_text()) for p in sorted(self.ops_dir.glob("*/meta.json"))}
        sp = torch.load(self.ops_dir / "specimens.pt") if (self.ops_dir / "specimens.pt").exists() else {"gt": {}}
        self.gt = {k: v.numpy() for k, v in sp["gt"].items()}
        self._A, self._Y, self._cols, self.last = {}, {}, {}, {}
        self.ops()

    # --- registry -----------------------------------------------------------------------
    def ops(self):
        """Print the registered operators."""
        print(f"{'operator':22s} {'coder':9s} {'light':>6s} {'A':>14s}  built / registered")
        for n, m in self.meta.items():
            print(f"{n:22s} {m['coder']:9s} {m['light_kept']:6.3f} {str(tuple(m['A_shape'])):>14s}  "
                  f"{m.get('build_secs') or '-'}s / {m['registered']}")
        print(f"specimens: {list(self.gt)}")

    def A(self, op):
        """The operator as a JAX array (loaded once, ~5.4 GB RAM each; lab.free(op) to drop)."""
        if op not in self._A:
            t = time.time()
            a = torch.load(_ROOT / self.meta[op]["A"], map_location="cpu", mmap=True).numpy()
            self._A[op] = snp.array(a)
            print(f"[reconlab] loaded A[{op}] {a.shape} in {time.time() - t:.0f}s")
        return self._A[op]

    def free(self, op=None):
        for k in ([op] if op else list(self._A)):
            self._A.pop(k, None)

    def _Yclean(self, op, specimen, measure):
        if measure == "grid":
            return np.asarray(self.A(op) @ snp.array(self.gt[specimen].reshape(-1)))
        if op not in self._Y:
            self._Y[op] = torch.load(self.ops_dir / op / "Y.pt")
        return self._Y[op][specimen].numpy()

    def scale(self, op, ref):
        """Equal-laser-power factor: this coder's light kept relative to the reference coder's."""
        return self.meta[op]["light_kept"] / self.meta[ref]["light_kept"]

    def measure(self, op, specimen="neural_cell", snr=25.0, seed=0, power=False, ref=None, measure="fine"):
        """Noisy measurement y (numpy) and the operator scale to solve with."""
        y = self._Yclean(op, specimen, measure)
        s = self.scale(op, ref) if (power and ref and ref != op) else 1.0
        yref = self._Yclean(ref, specimen, measure) if (power and ref and ref != op) else y
        sigma = float(yref.mean()) * 10 ** (-snr / 10)
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = torch.randn(y.shape, generator=g).numpy()
        return s * y + sigma * n, s

    # --- reconstruction -----------------------------------------------------------------
    def _prior(self, preset, over):
        cfg = dict(PRESETS.get(preset, {})) if preset else {}
        cfg.update(over); return cfg

    def solve(self, op, specimen="neural_cell", snr=25.0, seed=0, preset="dncnn", power=False, ref=None,
              patterns=None, measure="fine", return_hist=False, quiet=False, **over):
        """One reconstruction -> (estimate numpy (rz,rx,ry), metrics dict[, history])."""
        gt = self.gt[specimen]; shape = gt.shape
        y, s = self.measure(op, specimen, snr, seed, power, ref, measure)
        rows_per = int(np.prod(self.meta[op]["meas_shape"][1:]))
        nm = None if patterns is None else int(patterns) * rows_per
        prior = self._prior(preset, over)
        t = time.time()
        out = run_pnp(self.A(op), snp.array(y), obj_shape=shape, gt=snp.array(gt), n_measure=nm,
                      return_hist=return_hist, phi_scale=s, **prior)
        est = np.asarray(out[0]); m = score(gt, est)
        m.update(secs=round(time.time() - t, 1), scale=s)
        if specimen == "beads":
            br = bead_report(est, BEADS, mirror_plane)
            m["ghost"] = float(np.mean([b["ghost_ratio"] for b in br])); m["beads"] = br
        self.last[op] = est
        if not quiet:
            print(f"[recon] {op:18s} {specimen} snr={snr:g} {'power' if power else 'dose'} {preset or 'custom'} "
                  f"-> PSNR {m['psnr']:.2f} dB  localDepthErr {m['local_depth_err']:.2f}  shape {m['shape']:.2f}"
                  + (f"  ghost {m['ghost']:.2f}" if "ghost" in m else "") + f"  ({m['secs']}s)")
        return (est, m, out[2]) if return_hist else (est, m)

    def recon(self, op, specimen="neural_cell", show=True, save=None, **kw):
        """Reconstruct and show the go-to view (MIPs vs GT + axial profile)."""
        est, m = self.solve(op, specimen, **kw)
        if show:
            gt = self.gt[specimen]
            emit(viz.dashboard(gt, est, name=op, axial=(axial_profile(gt), axial_profile(est)),
                               title=f"{op} | {specimen} | PSNR {m['psnr']:.2f} dB"), save)
        return est, m

    def compare(self, ops, specimen="neural_cell", power=False, ref=None, oracle=True, show=True, save=None, **kw):
        """Same specimen, noise draw and prior for every operator; MIPs + axial profiles + table."""
        ref = ref or ops[0]; gt = self.gt[specimen]; ests, rows = {}, {}
        for op in ops:
            ests[op], rows[op] = self.solve(op, specimen, power=power, ref=ref, **kw)
        if oracle:
            rows["depth-blind oracle"] = score(gt, np.clip(depth_blind_oracle(gt), 0, 1))
        print(f"\n  {'':20s} {'PSNR':>7s} {'localDepthErr':>13s} {'shape':>6s} {'ghost':>6s}")
        for k, m in rows.items():
            print(f"  {k:20s} {m['psnr']:7.2f} {m['local_depth_err']:13.2f} {m['shape']:6.2f} "
                  f"{m['ghost'] if 'ghost' in m else float('nan'):6.2f}")
        if show:
            emit(viz.show_mips({"GT": gt, **ests}, title=f"{specimen} | {'equal power' if power else 'equal dose'}"), save)
            emit(viz.overlay_axial_profile(axial_profile(gt), {k: axial_profile(v) for k, v in ests.items()},
                                           title=specimen), save and str(Path(save).with_suffix("")) + "_profile.png")
        return ests, rows

    # --- views --------------------------------------------------------------------------
    def _vols(self, ops, specimen, **kw):
        ops = [ops] if isinstance(ops, str) else ops
        return {"GT": self.gt[specimen], **{op: self.solve(op, specimen, **kw)[0] for op in ops}}

    def mips(self, ops, specimen="neural_cell", save=None, **kw):
        emit(viz.show_mips(self._vols(ops, specimen, **kw), title=specimen), save)

    def slices(self, ops, specimen="neural_cell", save=None, **kw):
        emit(viz.show_slices(self._vols(ops, specimen, **kw)), save)

    def error(self, op, specimen="neural_cell", save=None, **kw):
        est, _ = self.solve(op, specimen, **kw)
        emit(viz.show_error(self.gt[specimen], est, name=op), save)

    def dashboard(self, op, specimen="neural_cell", save=None, **kw):
        self.recon(op, specimen, save=save, **kw)

    def converge(self, op, specimen="neural_cell", save=None, **kw):
        _, _, hist = self.solve(op, specimen, return_hist=True, **kw)
        if hist is None:
            print("(solver exposed no iteration history)"); return
        emit(viz.show_convergence(hist, title=f"ADMM convergence ({op})"), save)

    def beads(self, ops, save=None, **kw):
        ops = [ops] if isinstance(ops, str) else ops
        ests, rows = self.compare(ops, specimen="beads", show=False, **kw)
        for op in ops:
            print(f"  {op}: " + "  ".join(f"z{b['true_z']}->{b['peak_z']} ghost {b['ghost_ratio']:.2f}"
                                          for b in rows[op]["beads"]))
        emit(viz.show_mips({"GT": self.gt["beads"], **ests}, title="beads (mirror pairs)"), save)
        return rows

    def measurements(self, op, specimen="neural_cell", snr=None, n=64, save=None):
        """The detector frames (first n patterns) the reconstruction sees; snr=None -> clean."""
        y = self._Yclean(op, specimen, "fine") if snr is None else self.measure(op, specimen, snr)[0]
        T, a, b = self.meta[op]["meas_shape"]
        emit(viz.show_measurements(y.reshape(T, a, b)[:n], (min(n, T), a, b),
                                   title=f"{op} | {specimen} | {'clean' if snr is None else f'SNR {snr:g}'}"), save)

    def depth_corr(self, op, save=None):
        """Depth-column correlation of A (anti-diagonal = the +d/-d degeneracy)."""
        emit(viz.show_depth_corr(np.asarray(self.A(op)), tuple(self.meta[op]["recon_shape"]),
                                 title=f"depth-column correlation ({op})"), save)

    # --- validated metric ---------------------------------------------------------------
    def columns(self, op):
        """A's columns at the metric's sample voxels (16 lateral x all depths), numpy (M, 16*rz)."""
        if op not in self._cols:
            rz, rx, ry = self.meta[op]["recon_shape"]
            idx = [z * rx * ry + x * ry + y for (x, y) in LAT for z in range(rz)]
            a = torch.load(_ROOT / self.meta[op]["A"], map_location="cpu", mmap=True)
            self._cols[op] = a[:, idx].float().numpy()
        return self._cols[op]

    def predict(self, ops, specimen="neural_cell", snrs=(10, 15, 20, 25, 30), power=False, ref=None):
        ops = [ops] if isinstance(ops, str) else ops; ref = ref or ops[0]
        g = self.gt[specimen].reshape(-1); p0 = float(g.var()); peak = float(g.max() - g.min())
        print(f"{'operator':22s} " + " ".join(f"SNR{s:>4g}" for s in snrs)
              + f"   predicted PSNR, {'equal power vs ' + ref if power else 'equal dose'}")
        out = {}
        for op in ops:
            s = self.scale(op, ref) if power else 1.0
            sv = np.linalg.svd(self.columns(op) * s, compute_uv=False)
            ym = float(self._Yclean(ref if power else op, specimen, "fine").mean())
            out[op] = [10 * math.log10(peak ** 2 / float(np.mean(1.0 / (sv ** 2 / (ym * 10 ** (-q / 10)) ** 2 + 1.0 / p0))))
                       for q in snrs]
            print(f"{op:22s} " + " ".join(f"{v:7.2f}" for v in out[op]))
        return out
