# Dataset Card — Kvasir-Capsule (CrohnTracker mapping)

## Summary
CrohnTracker uses the public **Kvasir-Capsule** capsule endoscopy dataset as the primary source for training and evaluating a 4-class lesion taxonomy.

## Target classes
- `normal`
- `erythema`
- `erosion`
- `ulcer`

Other Kvasir-Capsule findings are remapped to `other_unmapped` and excluded from training by default.

## Splits
Train / validation / test splits are created at the **video** level (`ml/data/split_by_video.py`) to prevent frame leakage from the same sequence into multiple splits.

## License & access
Obtain Kvasir-Capsule from the official Simula / OSF distribution and follow the authors’ license terms. This repository does **not** redistribute the full dataset.

## Citation
Smedsrud et al., *Kvasir-Capsule, a video capsule endoscopy dataset*, Scientific Data, 2021.

## Demo fixtures
`data/demo` contains clearly labeled **non-medical placeholder** images for runnable UI demos. They are not training data and must not be presented as clinical imagery.
