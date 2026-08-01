"""
Torch-free reimplementation of the crop/pathology scanning + split logic from
gacl/image_dataset.py, so we can run genuine feature-extraction experiments on
My_Data.zip in an environment without PyTorch installed. The class map and
splitting rules are copied verbatim from gacl/image_dataset.py for consistency
with the rest of the codebase.
"""
import os, hashlib
from dataclasses import dataclass
import numpy as np

CLASS_TO_CROP_PATHOLOGY = {
    "Anthracnose on Cotton":     ("Cotton",    "Anthracnose",       "confirmed"),
    "Becterial Blight in Rice":  ("Rice",      "Bacterial Blight",  "confirmed"),
    "bollworm on Cotton":        ("Cotton",    "Bollworm",          "confirmed"),
    "Brownspot":                 ("Rice",      "Brown Spot",        "inferred_moderate"),
    "Common_Rust":               ("Maize",     "Common Rust",       "inferred_high"),
    "Cotton Aphid":              ("Cotton",    "Aphid",             "confirmed"),
    "Flag Smut":                 ("Wheat",     "Flag Smut",         "confirmed"),
    "Gray_Leaf_Spot":            ("Maize",     "Gray Leaf Spot",    "confirmed"),
    "Healthy Maize":             ("Maize",     "Healthy",           "confirmed"),
    "Healthy Wheat":             ("Wheat",     "Healthy",           "confirmed"),
    "Healthy cotton":            ("Cotton",    "Healthy",           "confirmed"),
    "Mosaic sugarcane":          ("Sugarcane", "Mosaic",            "confirmed"),
    "RedRot sugarcane":          ("Sugarcane", "Red Rot",           "confirmed"),
    "Rice Blast":                ("Rice",      "Blast",             "confirmed"),
    "Sugarcane Healthy":         ("Sugarcane", "Healthy",           "confirmed"),
    "Wheat Brown leaf rust":     ("Wheat",     "Brown Leaf Rust",   "confirmed"),
    "Wheat black rust":          ("Wheat",     "Black Rust",        "confirmed"),
    "cotton mealy bug":          ("Cotton",    "Mealy Bug",         "confirmed"),
    "cotton whitefly":           ("Cotton",    "Whitefly",          "confirmed"),
    "maize ear rot":             ("Maize",     "Ear Rot",           "confirmed"),
    "maize fall armyworm":       ("Maize",     "Fall Armyworm",     "confirmed"),
    "maize stem borer":          ("Maize",     "Stem Borer",        "confirmed"),
}

CROP_CLASSES_REAL = ["Cotton", "Rice", "Maize", "Wheat", "Sugarcane"]
PATHOLOGY_CLASSES_REAL = sorted({v[1] for v in CLASS_TO_CROP_PATHOLOGY.values()})
AUGMENTATION_MARKERS = ["zoom_", "contrast_", "rotozoom", "translation_", "rotate", "flip", "brightness"]


def _looks_augmented(filename: str) -> bool:
    name = filename.lower()
    return any(marker in name for marker in AUGMENTATION_MARKERS)


@dataclass
class RealImageRecord:
    path: str
    folder: str
    crop: str
    pathology: str
    confidence: str
    is_augmented_variant: bool
    file_hash: str


def scan_real_image_root(root):
    records = []
    seen_hashes = set()
    for folder_name in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        mapping = CLASS_TO_CROP_PATHOLOGY.get(folder_name)
        if mapping is None:
            continue
        crop, pathology, confidence = mapping
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(folder_path, fname)
            try:
                content = open(fpath, "rb").read()
            except OSError:
                continue
            h = hashlib.md5(content).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            records.append(RealImageRecord(
                path=fpath, folder=folder_name, crop=crop, pathology=pathology,
                confidence=confidence, is_augmented_variant=_looks_augmented(fname),
                file_hash=h,
            ))
    return records


def split_records(records, val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    by_class = {}
    for r in records:
        by_class.setdefault(r.pathology, []).append(r)

    train, val, test = [], [], []
    for pathology, recs in by_class.items():
        clean = [r for r in recs if not r.is_augmented_variant]
        augmented = [r for r in recs if r.is_augmented_variant]
        idx = rng.permutation(len(clean))
        clean_shuffled = [clean[i] for i in idx]

        n = len(clean_shuffled)
        n_val = max(1, int(round(n * val_frac))) if n >= 4 else 0
        n_test = max(1, int(round(n * test_frac))) if n >= 4 else 0
        n_val, n_test = min(n_val, n // 3 if n >= 3 else 0), min(n_test, n // 3 if n >= 3 else 0)

        val.extend(clean_shuffled[:n_val])
        test.extend(clean_shuffled[n_val:n_val + n_test])
        train.extend(clean_shuffled[n_val + n_test:])
        train.extend(augmented)

    return {"train": train, "validation": val, "test": test}
