"""
Specimens on the analysis grid (32 x 96 x 96 at 0.32 um): stored neural / blood cell volumes and bead phantoms.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline.forward.common import bead_phantom

# recon-grid (rz, rx, ry) bead positions; mirror of plane k is 14-k (see mirror_plane)
BEADS = [(1, 10, 10), (13, 10, 38), (3, 38, 10), (11, 38, 38), (5, 24, 10), (9, 24, 38)]
# held-out bead phantom used ONLY for tuning the sparsity prior (never scored): different positions,
# count and lateral layout from BEADS, with the same kind of mirror pairs (z <-> 14 - z).
BEADS_VAL = [(2, 8, 24), (12, 8, 30), (4, 30, 40), (10, 30, 8), (6, 42, 20), (8, 42, 44), (0, 18, 18), (14, 16, 44)]
STORED = {"neural_cell": "backup_fyp/D2NN/X_neural_cell.pt",
          "blood_cell": "backup_fyp/D2NN/X_blood_cell.pt"}
NAMES = ["neural_cell", "blood_cell", "beads", "beads_val"]


def mirror_plane(rz: int) -> int:
    """Recon plane mirrored about the focal plane (focus_idx 15 on the 32-plane grid -> recon 7.5)."""
    return 14 - rz


def stored_specimen(path, point, nz, N):
    """Stored specimen (32x128x128) area-resampled laterally to (nz, N, N), max-normalised."""
    raw = torch.load(path, map_location="cpu", mmap=True)
    x = raw[point].reshape(32, 128, 128).float().clone(); del raw
    x = F.adaptive_avg_pool3d(x[None, None], (nz, N, N))[0, 0]
    return x / (x.max() + 1e-12)


COHORT = [f"neural_cell_p{i}" for i in (64, 128, 192, 320, 448, 576, 704, 832)]


def make_specimen(name, cfg, root=".", point=0):
    """Fine (analysis-grid) object (nz, N, N) on the CPU."""
    N, nz = cfg.n_fine // cfg.bin_factor, cfg.nz
    if "_p" in name and name.rsplit("_p", 1)[0] in STORED:
        base, i = name.rsplit("_p", 1)
        return stored_specimen(Path(root) / STORED[base], int(i), nz, N)
    if name in ("beads", "beads_val"):
        return bead_phantom(nz, N, BEADS if name == "beads" else BEADS_VAL, patch=tuple(cfg.recon_patch), radius_um=0.8,
                            dz_um=cfg.dz_um, pitch_um=cfg.pitch_scalar, device="cpu")
    return stored_specimen(Path(root) / STORED[name], point, nz, N)
