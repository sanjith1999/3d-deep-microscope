# learned-pnp-prior

Neuron-trained 2D + 3D DnCNN inside PnP (`td2` / `td3` in `run_pnp`).
Training crops overlapping any test crop are dropped first (`scripts/denoise/screen_overlap.py`).

- `python -u scripts/denoise/train_denoiser.py --dim 2 --name nc2d` (and `--dim 3 --name nc3d`)
- `python -u scripts/recon/solver_lab.py test --configs base,t23n --val nc --seeds 0,1,2`

~30% lower depth error at full scan; 1/8 of the frames ~ baseline full scan.
