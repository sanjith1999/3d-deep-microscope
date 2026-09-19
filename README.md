# detector-na-sampling

NA {0.1, 0.2, 0.3} x pixels {12, 24, 48}^2 at equal measurement count (256 / 64 / 16 frames).

- `python -u scripts/recon/detector_study.py build | tune | test`

NA 0.3 + 48x48 at 16 frames beats NA 0.1 + 12x12 at 256 frames. Pixels alone don't help.
