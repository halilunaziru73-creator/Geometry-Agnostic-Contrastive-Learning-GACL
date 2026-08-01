# Geometry Agnostic Contrastive Learning (GACL): A Theoretical Framework and Baseline Evaluation Protocol for Agronomic Imaging

This repository contains the manuscript, reference implementation, reference dataset,
and reproducible figure-generation scripts for **GACL** — a hypergraph-transformer
architecture (HGAViT + GCATT + DHGNN + VLAE) proposed for cross-pathology,
cross-acquisition-condition contrastive transfer learning in agronomic imaging.

**Honesty note (carried over from the manuscript itself):** GACL is implemented and
internally coherent, but **not yet empirically validated** as a trained deep-learning
classifier. The real diagnostics in this repo (ROC, permutation importance, PCA,
confusion matrix) are run against classical baselines and a tabular reference
dataset — not against a trained GACL model — because no GPU/PyTorch environment was
available when this package was prepared. See `code/gacl/README_GACL.md` and
`image_experiment/README.md` for exactly what is and isn't validated.

## Figures

The three real, reproducible figures currently included in this package (from
`image_experiment/`, Section 3.14 of the manuscript — real hand-crafted-feature
classifiers trained on real field photographs):

![Geometry invariance schematic](image_experiment/fig_geometry_invariance_schematic.png)
**fig_geometry_invariance_schematic.png** — Architectural schematic illustrating GACL's geometry-invariance goal

![Real classifier comparison](image_experiment/fig_real_classifier_comparison.png)
**fig_real_classifier_comparison.png** — Performance comparison of classical classifiers (KNN, Logistic Regression, Random Forest, Gradient Boosting, SVM, MLP) trained on real hand-crafted image features

![Real confusion matrix](image_experiment/fig_real_confusion_matrix.png)
**fig_real_confusion_matrix.png** — Confusion matrix for the best-performing classical classifier on the real held-out test split

> **Note:** the project `README.md` (preserved below, and also at
> `README_ORIGINAL.md`) describes five additional figures (`p1_new_*.png`, Figures
> 46-50) that were part of an earlier manuscript revision's `figures/` folder. That
> folder was not present in the uploaded package, so those five images are not in
> this repository. They can be regenerated from `data/GACL_Data.xlsx` using the
> scripts in `reproduce/` — see the reproduction steps below.

---

# Paper 1 submission package — GACL: A Hypergraph-Transformer Architecture for Cross-Pathology Contrastive Transfer

## Contents
- `paper/Paper1_GACL_Architecture_CEA.docx` — manuscript, reformatted and expanded for *Computers and Electronics in Agriculture* (Elsevier). Now 20 pages (from 16), single-column, editable .docx.
- `paper/Paper1_Highlights.docx` — separate Highlights file (5 bullets, ≤85 characters each), as required at submission.
- `figures/` — the five new figures added in this revision (Figures 46-50 in the manuscript):
  - `p1_new_gradflow_schematic.png` — architectural gradient-flow diagram (Fig. 46)
  - `p1_new_complexity.png` — analytic HGAViT self-attention cost curve from Eq. 39 (Fig. 47)
  - `p1_new_roc.png` — real one-vs-rest ROC curves, RandomForest on GACL_Data.xlsx (Fig. 48)
  - `p1_new_permimportance.png` — real permutation feature importance (Fig. 49)
  - `p1_new_pca.png` — real PCA projection of the embedding/latent columns (Fig. 50)
- `code/` — the full GACL reference implementation (`gacl/hgavit.py`, `gcatt.py`, `dhgnn.py`, `vlae.py`, `losses.py`, `model.py`), training/evaluation entry points (`train_gacl.py`, `train_gacl_real_images.py`, `evaluate_gacl.py`, `check_learnable_structure.py`), and the shared `core/` analysis package referenced throughout the manuscript.
- `data/GACL_Data.xlsx` — the tabular reference dataset (`Measured_Data` sheet, 20,000 rows, 75 columns) used for Section 3.12's and Section 3.14's real diagnostics.
- `reproduce/` — the exact, standalone scripts used to regenerate every new figure from `data/GACL_Data.xlsx`:
  - `gen_p1_a.py` — fits and caches the RandomForestClassifier (run first)
  - `gen_p1_b.py` — ROC figure (Fig. 48)
  - `gen_p1_c.py` — PCA figure (Fig. 50)
  - `gen_p1_d.py` — analytic complexity figure (Fig. 47)
  - `gen_p1_e.py` — permutation importance figure (Fig. 49)
  - `gen_schematics_p1_p2.py` — the gradient-flow schematic (Fig. 46) and the companion Paper-2 architecture schematic

## Reproducing the new figures
```
cd code   # so that data/GACL_Data.xlsx paths resolve, or edit paths in the scripts
python3 ../reproduce/gen_p1_a.py
python3 ../reproduce/gen_p1_b.py
python3 ../reproduce/gen_p1_c.py
python3 ../reproduce/gen_p1_d.py
python3 ../reproduce/gen_p1_e.py
python3 ../reproduce/gen_schematics_p1_p2.py
```
Requires: pandas, numpy, scikit-learn, matplotlib, openpyxl, joblib (see `code/requirements.txt`).

## What changed in this revision
1. Reformatted to *Computers and Electronics in Agriculture*'s Guide for Authors: single-column layout, trimmed keyword list to 6, separate Highlights file, added Funding statement and Declaration of Generative AI use.
2. Expanded Related Work with a new §2.4 tracing each of GACL's four components to its methodological lineage.
3. Added a new §3.14 with three additional real diagnostics (ROC, permutation importance, PCA) reinforcing — with finer-grained evidence — the paper's existing, honestly reported null result on the tabular reference data.
4. Added a new Appendix B with three additional formal derivations (attention Jacobian, expanded VLAE KL term, numerical-stability note).
5. No performance claims were strengthened or altered: GACL remains explicitly implemented-but-empirically-unvalidated, consistent with the original manuscript's own framing.

## Post-submission formatting pass (this update)
- **Equation numbering fixed**: 13 equations (Eqs. 27–39, main text) had their equation-number label sitting in a separate, disconnected, left-aligned paragraph below the formula. Each number has been merged back onto the same centred line as its equation, matching the style already used for the Appendix B equations in this same manuscript.
- **Reference DOIs completed**: 14 reference-list entries were missing a resolvable link (conference papers, MDPI/Elsevier journal articles, and arXiv preprints). Each has been checked against the publisher/Crossref/arXiv record and given its correct `https://doi.org/...` link. Two entries (Higgins et al. 2017; Quiñonero-Candela et al. 2009) have no DOI to give — these are marked "No DOI" with the reason (ICLR proceedings paper / book) rather than left blank or given a placeholder link.
