#!/usr/bin/env python3
"""
Leakage screen: drop training crops that share voxels with any test, showcase or validation crop.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import torch
from pipeline.specimens import STORED

PROTECT = {"neural_cell": [0, 64, 128, 192, 320, 448, 576, 704, 832, 384, 973], "blood_cell": [0]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stack", default="neural_cell")
    ap.add_argument("--r_thr", type=float, default=0.9); ap.add_argument("--min_frac", type=float, default=0.02)
    ap.add_argument("--peaks", type=int, default=5); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    X = torch.load(_ROOT / STORED[a.stack], map_location="cpu", mmap=True)
    n = X.shape[0]; S = (32, 128, 128); P = (64, 256, 256)
    vol = lambda i: X[i].reshape(S).float().to(dev)
    def spec(v):
        z = (v - v.mean()) / (v.std() + 1e-8)
        return torch.fft.rfftn(z, s=P)
    prot = PROTECT[a.stack]; Fp = {p: spec(vol(p)) for p in prot}; Vp = {p: vol(p) for p in prot}
    excluded, train = {}, []
    for i in range(n):
        if i in prot:
            continue
        vi = vol(i); Fi = spec(vi); hit = None
        for p in prot:
            cc = torch.fft.irfftn(Fp[p] * Fi.conj(), s=P)                 # cc[s] = sum_r a(r + s) b(r)  (a = protected)
            top = torch.topk(cc.flatten(), a.peaks).indices
            for t in top.tolist():
                s = [int(c) for c in torch.unravel_index(torch.tensor(t), P)]
                s = [c - P[d] if c > P[d] // 2 else c for d, c in enumerate(s)]   # signed shift
                sl_a = tuple(slice(max(0, sd), S[d] + min(0, sd)) for d, sd in enumerate(s))
                sl_b = tuple(slice(max(0, -sd), S[d] - max(0, sd)) for d, sd in enumerate(s))
                A, B = Vp[p][sl_a].flatten(), vi[sl_b].flatten()
                frac = A.numel() / vi.numel()
                if frac < a.min_frac or A.numel() < 100:
                    continue
                r = float(torch.corrcoef(torch.stack([A, B]))[0, 1])
                if r > a.r_thr:
                    hit = {"protected": p, "shift_zxy": s, "overlap_frac": round(frac, 4), "r": round(r, 4)}; break
            if hit:
                break
        if hit:
            excluded[i] = hit
        else:
            train.append(i)
        if i % 128 == 0:
            print(f"[overlap] {a.stack}: {i}/{n}  excluded so far {len(excluded)}", flush=True)
    out = _ROOT / "results/denoisers"; out.mkdir(parents=True, exist_ok=True)
    (out / f"overlap_{a.stack}.json").write_text(json.dumps({"protected": prot, "train": train, "excluded": excluded,
                                                              "r_thr": a.r_thr, "min_frac": a.min_frac}, indent=1))
    print(f"[overlap] {a.stack}: {len(train)} train crops, {len(excluded)} excluded (overlap with a protected crop)")


if __name__ == "__main__":
    main()
