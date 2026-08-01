import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df1 = pd.read_excel("N_GACL/N_GACL/GACL_Data.xlsx", sheet_name="Measured_Data")
num_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
feat_cols = [c for c in num_cols if c not in ('sample_id',)]
X = df1[feat_cols].fillna(0.0)
y = df1['pathology'].astype(str)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=2)
clf.fit(Xtr, ytr)
joblib.dump((clf, Xtr, Xte, ytr, yte, feat_cols), "p1_rf.joblib")
print("fit done", clf.score(Xte, yte))
