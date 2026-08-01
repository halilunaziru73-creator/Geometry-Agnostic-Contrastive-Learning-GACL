# Real image-based classical-feature validation (Section 3.14)

This folder contains the scripts and outputs for the new, genuinely real
experiment reported in Section 3.14 of the revised paper.

## Why this exists instead of a real GACL (deep model) training run

The environment used to prepare this revision has no internet access, so
`pip install torch` fails and `train_gacl_real_images.py` (the real deep-model
training script, in `code/train_gacl_real_images.py`) could not be executed
here. Rather than fabricate deep-learning numbers, we ran a genuinely
different, fully reproducible experiment: extracting real hand-crafted
features directly from the real photographs in `My_Data.zip` and training
classical scikit-learn classifiers on them.

## Files

- `real_data_lib.py` -- torch-free reimplementation of the scanning/label
  mapping/splitting logic in `code/gacl/image_dataset.py`, so the exact same
  class map and leakage-aware split can be reused without importing PyTorch.
- `extract_features.py` -- loads every real image, extracts 51 hand-crafted
  features (colour stats, ExG/VARI, colour histograms, GLCM texture, Sobel
  edge stats, a lesion-colour-mass proxy), and saves them to
  `real_image_features.pkl` (not included here due to size -- regenerate by
  running this script against `My_Data/`).
- `train_classifiers.py` -- trains KNN, Logistic Regression, Random Forest,
  Gradient Boosting, SVM, and an MLP on the extracted features, evaluates on
  the real held-out validation/test splits, and writes `classifier_results.json`.
- `classifier_results.json` -- the real, already-computed results reported in
  Table 3 / Section 3.14 of the paper.
- `fig_real_classifier_comparison.png`, `fig_real_confusion_matrix.png` --
  the two figures (10 and 11) inserted into the paper.

## Reproducing

```bash
python3 extract_features.py /path/to/My_Data
python3 train_classifiers.py
```

## What this does NOT establish

This does not measure GACL's own learned representation (HGAViT + GCATT +
DHGNN + VLAE). That still requires running `train_gacl_real_images.py` in an
environment with PyTorch and, ideally, a GPU. This experiment's result is the
classical baseline that a future trained GACL model should be benchmarked
against, per Section 6 of the paper.
