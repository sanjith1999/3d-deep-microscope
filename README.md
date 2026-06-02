# speckle-intensity-pdf

Plate design from the illumination only: axial decorrelation + intensity PDF / contrast + smoothness (`coder/losses.py`).

- `python scripts/coder/train.py`
- `python scripts/coder/e1_symmetry.py --designed <run>/coder_final.pt`

Breaks the +d/-d symmetry, but at physical neuron pitch (0.32 um) barely beats a random plate in reconstruction.
