# FRIS

3D fluorescence imaging with a diffractive phase plate in the illumination path and one camera.
Angle-scanned plane wave -> phase plate -> 4F relay (NA 0.8) -> depth-varying speckle; volume recovered with PnP.

Code branches:

- `base-mdl` - forward model, operator, PnP solver, compression runs (main results)
- `experiment/speckle-intensity-pdf` - surrogate-loss plate design
- `experiment/dmd-baseline` - DMD comparisons
- `experiment/model-mismatch` - coarse operator ablation
- `experiment/learned-pnp-prior` - trained denoisers in PnP
- `experiment/unrolled-modl` - MoDL / HQS
- `experiment/detector-na-sampling` - detection NA vs pixels vs frames
- `experiment/detector-aware-plate` - plate trained through the detector

Paper branches:

- `paper/manuscript` - original manuscript and extended experimental record
- `paper/cvpr/init-draft` - initial CVPR manuscript and supplement
- `paper/cvpr/struct-amb` - CVPR draft focused on structural ambiguity
- `paper/cvpr/dec-agn-depth` - CVPR draft on decoder-free depth encoding
- `paper/cvpr/depth-sep` - CVPR drafts on learned depth separation and axial
  structure recovery
- `paper/cvpr/axial-rec` - current CVPR draft on optical learning for axial
  recovery, compared with established objectives and end-to-end training

Each CVPR branch has `main.tex` and `supplement.tex` at the root. The original
research branch has separate `manuscript/` and `experiments/` projects. Each
branch includes build instructions and self-contained figures and references.

Tags mark successive CVPR drafts: `cvpr-v0.1` on `paper/cvpr/dec-agn-depth`,
`cvpr-v0.2` and `cvpr-v0.3` on `paper/cvpr/depth-sep`, and `cvpr-v0.4` to
`cvpr-v0.7` on `paper/cvpr/axial-rec`.
