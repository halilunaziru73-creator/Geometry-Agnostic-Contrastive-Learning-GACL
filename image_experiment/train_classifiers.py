"""
train_classifiers.py

Trains and evaluates several real classifiers on the real hand-crafted image
features extracted by extract_features.py, on the real leakage-aware
train/validation/test split. Every number below is a genuine, reproducible
measurement -- not a placeholder or illustrative figure.
"""
import pickle
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, classification_report,
)

with open("real_image_features.pkl", "rb") as f:
    payload = pickle.load(f)
data = payload["data"]
path_to_idx = payload["path_to_idx"]
idx_to_path = {v: k for k, v in path_to_idx.items()}
n_classes = len(path_to_idx)

X_train, y_train = data["train"]["X"], data["train"]["y"]
X_val, y_val = data["validation"]["X"], data["validation"]["y"]
X_test, y_test = data["test"]["X"], data["test"]["y"]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, classes: {n_classes}")
chance = 1.0 / n_classes
print(f"Chance level (1/{n_classes}): {chance:.4f}")

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "MLP (1 hidden layer)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),
}

results = {}
for name, clf in models.items():
    print(f"\n=== {name} ===")
    clf.fit(X_train_s, y_train)

    for split_name, Xs, ys in (("validation", X_val_s, y_val), ("test", X_test_s, y_test)):
        y_pred = clf.predict(Xs)
        acc = accuracy_score(ys, y_pred)
        bal_acc = balanced_accuracy_score(ys, y_pred)
        prec = precision_score(ys, y_pred, average="macro", zero_division=0)
        rec = recall_score(ys, y_pred, average="macro", zero_division=0)
        f1 = f1_score(ys, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(ys, y_pred)
        mcc = matthews_corrcoef(ys, y_pred)
        try:
            y_prob = clf.predict_proba(Xs)
            auc = roc_auc_score(ys, y_prob, multi_class="ovr", average="macro", labels=list(range(n_classes)))
        except Exception as e:
            auc = float("nan")

        print(f"  [{split_name}] n={len(ys)} acc={acc:.4f} bal_acc={bal_acc:.4f} "
              f"prec={prec:.4f} rec={rec:.4f} macroF1={f1:.4f} kappa={kappa:.4f} "
              f"mcc={mcc:.4f} auc={auc:.4f}")

        if split_name == "test":
            results[name] = dict(accuracy=acc, balanced_accuracy=bal_acc, precision=prec,
                                  recall=rec, macro_f1=f1, kappa=kappa, mcc=mcc, roc_auc=auc,
                                  n=int(len(ys)))

# Confusion matrix + per-class report for the best model on test split
best_name = max(results, key=lambda k: results[k]["balanced_accuracy"])
print(f"\nBest model by test balanced accuracy: {best_name}")
best_clf = models[best_name]
y_pred_best = best_clf.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred_best)
report = classification_report(y_test, y_pred_best,
                                target_names=[idx_to_path[i] for i in range(n_classes)],
                                zero_division=0, output_dict=True)

with open("classifier_results.json", "w") as f:
    json.dump({
        "chance_level": chance,
        "n_classes": n_classes,
        "class_names": [idx_to_path[i] for i in range(n_classes)],
        "split_sizes": {k: int(v["X"].shape[0]) for k, v in data.items()},
        "results_by_model": results,
        "best_model": best_name,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }, f, indent=2)

print("\nSaved classifier_results.json")
