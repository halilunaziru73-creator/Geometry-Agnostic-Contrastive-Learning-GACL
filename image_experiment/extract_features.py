"""
extract_features.py

Real feature extraction from the real crop-disease photographs in My_Data.zip,
using only numpy / PIL / scikit-image (no PyTorch -- see NOTE below).

NOTE ON WHY THIS EXISTS
-----------------------
The repo's actual deep-learning architecture (gacl/hgavit.py, gcatt.py,
dhgnn.py, vlae.py, model.py) and its real-image training entry point
(train_gacl_real_images.py) require PyTorch. This sandbox has no internet
access, so `pip install torch` fails and that script cannot be executed here
-- exactly the limitation the paper itself already documents for the tabular
pseudo-patch run. Rather than fabricate deep-learning numbers, this script
runs a genuinely new, honest experiment: extracting real hand-crafted
features directly from the real photographs (not the earlier simulated
tabular dataset) and training real classical classifiers on them, with a
proper leakage-aware, class-stratified split reused from gacl/image_dataset.py.
"""
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel

from real_data_lib import scan_real_image_root, split_records, PATHOLOGY_CLASSES_REAL

IMAGE_SIZE = 128


def load_image(path):
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def extract_features(arr):
    """Real, deterministic feature vector for one RGB image array in [0,255]."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = rgb2gray(arr / 255.0)

    feats = []
    # --- Colour statistics (mean/std per channel + normalised chromaticity) ---
    for ch in (r, g, b):
        feats += [ch.mean(), ch.std()]
    total = r.mean() + g.mean() + b.mean() + 1e-9
    feats += [r.mean() / total, g.mean() / total, b.mean() / total]

    # --- Vegetation-index-style proxies (as used elsewhere in this repo) ---
    exg = 2 * g.mean() - r.mean() - b.mean()
    vari = (g.mean() - r.mean()) / (g.mean() + r.mean() - b.mean() + 1e-6)
    feats += [exg, vari]

    # --- Colour histograms (coarse, 8 bins per channel) ---
    for ch in (r, g, b):
        hist, _ = np.histogram(ch, bins=8, range=(0, 255))
        feats += (hist / hist.sum()).tolist()

    # --- GLCM texture features (grey-level co-occurrence matrix) ---
    gray_u8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_u8, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"):
        feats += graycoprops(glcm, prop).mean(axis=1).tolist()

    # --- Edge / shape proxy statistics ---
    edges = sobel(gray)
    feats += [edges.mean(), edges.std(), (edges > edges.mean() + edges.std()).mean()]

    # --- Simple lesion-colour-mass proxy: fraction of pixels far from typical green ---
    green_dist = np.sqrt((r - 60) ** 2 + (g - 130) ** 2 + (b - 40) ** 2)
    feats += [float((green_dist > 90).mean())]

    return np.array(feats, dtype=np.float64)


def build_dataset(data_root):
    records = scan_real_image_root(data_root)
    splits = split_records(records)
    path_to_idx = {p: i for i, p in enumerate(PATHOLOGY_CLASSES_REAL)}

    out = {}
    for split_name, recs in splits.items():
        X, y, folders, confidences = [], [], [], []
        for r in recs:
            try:
                arr = load_image(r.path)
            except Exception as e:
                print(f"  [skip] failed to load {r.path}: {e}")
                continue
            X.append(extract_features(arr))
            y.append(path_to_idx[r.pathology])
            folders.append(r.folder)
            confidences.append(r.confidence)
        out[split_name] = dict(X=np.stack(X), y=np.array(y), folders=folders, confidences=confidences)
        print(f"{split_name}: {out[split_name]['X'].shape[0]} images, "
              f"feature dim {out[split_name]['X'].shape[1]}")
    return out, path_to_idx, records


if __name__ == "__main__":
    import sys, pickle
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/work/My_Data"
    data, path_to_idx, records = build_dataset(data_root)
    with open("real_image_features.pkl", "wb") as f:
        pickle.dump({"data": data, "path_to_idx": path_to_idx, "n_records": len(records)}, f)
    print("Saved real_image_features.pkl")
