# model-mismatch

Patch (sum of fine columns) vs point-sampled vs coarse-simulated operator, same measurements.

- `python -u scripts/recon/mismatch_ablation.py all --op random1L_gap66 --label one_plate`
- `python -u scripts/recon/mismatch_report.py --dir results/paper/mismatch/one_plate`

Patch wins.
