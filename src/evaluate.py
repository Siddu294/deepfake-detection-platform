"""Evaluate the trained checkpoint on a separate image test directory.

Expected structure: test_dataset/fake/* and test_dataset/real/*.
Only images are evaluated here; video evaluation should use video-level labels
and must ensure source videos are absent from training/validation.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.inference.predictor import DeepfakePredictor, IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test_dataset", help="Folder containing fake/ and real/ image subfolders")
    parser.add_argument("--model", default="models/baseline_model.pth")
    args = parser.parse_args()
    root = Path(args.data)
    samples = []
    for class_name, label in (("fake", 0), ("real", 1)):
        folder = root / class_name
        if not folder.is_dir():
            raise SystemExit(f"Missing class folder: {folder}")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    if not samples:
        raise SystemExit(f"No supported image files found under {root}/fake and {root}/real")
    predictor = DeepfakePredictor(args.model)
    y_true, y_pred, fake_scores = [], [], []
    skipped = 0
    for path, label in samples:
        try:
            result = predictor.predict_image(path)
        except (ValueError, OSError, UnidentifiedImageError) as exc:
            print(f"Skipping unreadable file {path}: {exc}")
            skipped += 1
            continue
        y_true.append(label)
        y_pred.append(0 if result["prediction"] == "FAKE" else 1)
        fake_scores.append(result["fake_probability"])
    if not y_true:
        raise SystemExit("No readable test images were evaluated.")
    print(f"Evaluated: {len(y_true)} | Skipped: {skipped}")
    print("Confusion matrix (rows=true [fake, real], columns=predicted [fake, real]):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["FAKE", "REAL"], zero_division=0, digits=4))
    if len(set(y_true)) == 2:
        try:
            print(f"ROC-AUC (fake probability): {roc_auc_score(y_true, fake_scores):.4f}")
        except ValueError as exc:
            print(f"ROC-AUC unavailable: {exc}")
    else:
        print("ROC-AUC unavailable: the test set must contain both classes.")


if __name__ == "__main__":
    main()
