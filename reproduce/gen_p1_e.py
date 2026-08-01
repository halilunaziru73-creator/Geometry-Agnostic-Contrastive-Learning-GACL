import numpy as np, matplotlib, joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

clf, Xtr, Xte, ytr, yte, feat_cols = joblib.load("p1_rf.joblib")
# subsample test set for tractable permutation importance
rng = np.random.RandomState(0)
idx = rng.choice(len(Xte), size=min(1200, len(Xte)), replace=False)
Xte_s = Xte.iloc[idx]; yte_s = yte.iloc[idx]
pim = permutation_importance(clf, Xte_s, yte_s, n_repeats=3, random_state=42, n_jobs=2)
order = np.argsort(pim.importances_mean)[::-1][:15]
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
plt.figure(figsize=(6,4.5))
plt.barh(np.array(feat_cols)[order][::-1], pim.importances_mean[order][::-1],
         xerr=pim.importances_std[order][::-1], color="#4C72B0")
plt.xlabel("Permutation importance (mean accuracy decrease)")
plt.title("Real permutation feature importance, top 15 features\n(RandomForest, GACL_Data.xlsx, 1,200-sample test subsample)")
plt.tight_layout(); plt.savefig("figs/p1_new_permimportance.png"); plt.close()
print("perm importance done")
