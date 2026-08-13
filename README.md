# FRIS - base model

- grids: analysis 32x96x96 @ 0.32 um, recon 16x48x48 @ 0.64 um
- build operator: `python -u scripts/recon/build_fast.py --coder random --seed 0 --n_layers 1 --name random1L_gap66`
- compression: `scripts/recon/compression_subsets.py select | tune | run | report`
- noise: q = 15 + 5 log10(256/k), tune only on validation specimens
- specimen volumes in `backup_fyp/D2NN/`, outputs in `results/`
