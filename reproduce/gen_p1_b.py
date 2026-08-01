import numpy as np, matplotlib, joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

clf, Xtr, Xte, ytr, yte, feat_cols = joblib.load("p1_rf.joblib")
classes = clf.classes_
yte_bin = label_binarize(yte, classes=classes)
proba = clf.predict_proba(Xte)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
plt.figure(figsize=(5.2,4.6))
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(yte_bin[:, i], proba[:, i])
    plt.plot(fpr, tpr, alpha=0.7, lw=1.3, label=f"{c} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Real one-vs-rest ROC, RandomForest on GACL_Data.xlsx\n(full 74-feature set, held-out test split)")
plt.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("figs/p1_new_roc.png"); plt.close()
print("roc done")
