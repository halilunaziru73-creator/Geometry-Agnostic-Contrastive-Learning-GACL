# Changes in this revision

This revision was prepared in a sandboxed environment with no internet access
and no PyTorch installation. Both of those constraints are the same ones the
original paper already documented for the deep-learning training scripts, and
they still apply here: `train_gacl_real_images.py` and `train_gacl.py`
(the actual GACL deep-model training scripts) could not be executed.

## What is genuinely new in this revision

A real, reproducible experiment (see `real_image_experiment/`) was run
directly against the 1,543 real crop-disease photographs in `My_Data.zip`:

- Hand-crafted features (colour, GLCM texture, vegetation-index proxies,
  edge statistics) were extracted from every real image.
- Six classical classifiers (KNN, Logistic Regression, Random Forest,
  Gradient Boosting, SVM, MLP) were trained and evaluated on a real,
  leakage-aware, stratified train/validation/test split reusing the exact
  label mapping and split logic already defined in `code/gacl/image_dataset.py`.
- The best model (a single-hidden-layer MLP) reached 64.4% raw accuracy /
  38.1% balanced accuracy / macro-F1 0.378 / Cohen's kappa 0.563 / macro
  ROC-AUC 0.882 on the held-out test split (19 pathology classes, 5.3%
  chance level).

This result is reported as new Section 3.14 in the paper, and the Abstract,
Contributions, Discussion, Limitations, Future Work, Conclusion, and Data
Availability sections were all updated to reference it consistently.

## What is explicitly NOT claimed

- No accuracy/F1/AUC number is reported anywhere in this paper for GACL's own
  four learned components (HGAViT, GCATT, DHGNN, VLAE). They remain
  implemented but untrained, because PyTorch could not be installed in this
  environment (no network access).
- The classical-feature result above is a baseline for a future trained GACL
  model to be compared against -- not a measurement of GACL itself.
- An earlier, unverified figure (33.8% balanced accuracy) that appeared in a
  prior version of this paper for a similar classical-feature check was not
  independently reproduced by us, and Section 3.13 now says so explicitly.
  The number in Section 3.14 (29.6% for Random Forest, 38.1% for the best
  model) is the one we actually ran and can vouch for.

## Next step, if you want a genuinely trained GACL model

Run `code/train_gacl_real_images.py` in an environment with PyTorch and,
ideally, a GPU (e.g. Google Colab). It already prints real accuracy, balanced
accuracy, macro-F1, macro-AUC, Cohen's kappa, and MCC on held-out validation
and test splits, benchmarked against the Section 3.14 classical result. Send
the output back and it can be written into the paper as the deep model's own,
genuinely measured result.
