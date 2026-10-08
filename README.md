# FRIS

3D fluorescence imaging with a diffractive phase plate in the illumination path and one camera.
Angle-scanned plane wave -> phase plate -> 4F relay (NA 0.8) -> depth-varying speckle; volume recovered with PnP.

Branches:
- `base-mdl` - forward model, operator, PnP solver, compression runs (main results)
- `experiment/speckle-intensity-pdf` - surrogate-loss plate design
- `experiment/dmd-baseline` - DMD comparisons
- `experiment/model-mismatch` - coarse operator ablation
- `experiment/learned-pnp-prior` - trained denoisers in PnP
- `experiment/unrolled-modl` - MoDL / HQS
- `experiment/detector-na-sampling` - detection NA vs pixels vs frames
- `experiment/detector-aware-plate` - plate trained through the detector
