# detector-aware-plate

Plate trained to decorrelate depth-stacked operator columns through the real detector, starting from the random plate.

- `python -u scripts/coder/train_detector_aware.py --na_em 0.1 --d 8 --frames patterns_d8.json --name da_na01_d8`
- `python -u scripts/recon/detector_study.py test --designed da_na01_d8:0.1:8:da_na01_d8.pt`

Beats random plates on every test crop at the NA 0.1 detector.
