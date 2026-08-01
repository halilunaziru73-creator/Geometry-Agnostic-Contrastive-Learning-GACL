import numpy as np, matplotlib, joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

clf, Xtr, Xte, ytr, yte, feat_cols = joblib.load("p1_rf.joblib")
df1 = pd.read_excel("N_GACL/N_GACL/GACL_Data.xlsx", sheet_name="Measured_Data")
emb_cols = [c for c in feat_cols if ('embedding' in c or 'latent' in c)]
y = df1['pathology'].astype(str)
Xs = StandardScaler().fit_transform(df1[emb_cols].fillna(0.0))
p2 = PCA(n_components=2, random_state=42).fit(Xs)
Z = p2.transform(Xs)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
plt.figure(figsize=(5.6,4.8))
for c in sorted(y.unique()):
    m = (y.values == c)
    plt.scatter(Z[m,0], Z[m,1], s=6, alpha=0.45, label=c)
plt.xlabel(f"PC1 ({p2.explained_variance_ratio_[0]*100:.1f}% var.)")
plt.ylabel(f"PC2 ({p2.explained_variance_ratio_[1]*100:.1f}% var.)")
plt.title("Real PCA projection of GACL_Data.xlsx's precomputed\nembedding_1..32 + latent_z1..16 columns, coloured by pathology")
plt.legend(fontsize=7, markerscale=2, ncol=2)
plt.tight_layout(); plt.savefig("figs/p1_new_pca.png"); plt.close()
print("pca done, n_embed_cols=", len(emb_cols))
