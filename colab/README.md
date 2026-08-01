# Running the real GACL deep-model training (Colab)

`GACL_Colab_Training.ipynb` installs PyTorch, uploads this project plus
`My_Data.zip`, and runs `code/train_gacl_real_images.py` on a free Colab GPU
(Runtime -> Change runtime type -> GPU).

It prints real accuracy, balanced accuracy, macro-F1, macro-AUC, Cohen's
kappa, and MCC on held-out validation/test splits, next to their chance-level
references -- the actual GACL result Section 3.14 of the paper is currently
missing, to be compared directly against the 64.4% / 38.1% classical
baseline already reported there.

Send the printed output back once you've run it, and it can be written into
the paper as GACL's own, genuinely trained result.
