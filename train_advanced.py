"""Reproducible ResNet-50 training with train/validation/test evaluation.

Expected dataset layout:
  test_dataset/fake/*
  test_dataset/real/*
Supported entries may be images or videos. Videos are represented by their first
readable frame, matching the original project behavior. This is frame-level image
classification, not temporal video learning.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CLASSES = {"fake": 0, "real": 1}
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path):
    if path.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
            if not ok or frame is None: raise ValueError("no readable first frame")
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    with Image.open(path) as im:
        return im.convert("RGB")


class MediaDataset(Dataset):
    def __init__(self, items, transform): self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            image = load_rgb(path)
        except Exception as exc:
            raise RuntimeError(f"Cannot decode dataset file {path}: {exc}") from exc
        return self.transform(image), label


def collect_split(root: Path, seed=SEED):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls, label in CLASSES.items():
        folder = root / cls
        if not folder.is_dir(): raise FileNotFoundError(f"Missing class folder: {folder}")
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        if len(files) < 3: raise ValueError(f"Need at least 3 files in {folder}; found {len(files)}")
        rng.shuffle(files)
        n = len(files); n_train = max(1, int(n * .8)); n_val = max(1, int(n * .1))
        if n_train + n_val >= n: n_train, n_val = n - 2, 1
        train.extend((p, label) for p in files[:n_train])
        val.extend((p, label) for p in files[n_train:n_train+n_val])
        test.extend((p, label) for p in files[n_train+n_val:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def evaluate(model, loader, device):
    model.eval(); ys, preds, fake_scores = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device)); probs = torch.softmax(logits, dim=1)
            ys.extend(y.tolist()); preds.extend(logits.argmax(1).cpu().tolist())
            fake_scores.extend(probs[:, 0].cpu().tolist())
    result = {"accuracy": float(accuracy_score(ys, preds)),
              "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
              "confusion_matrix": confusion_matrix(ys, preds, labels=[0, 1]).tolist(),
              "classification_report": classification_report(ys, preds, labels=[0, 1], target_names=["FAKE", "REAL"], output_dict=True, zero_division=0)}
    if len(set(ys)) == 2: result["roc_auc_fake"] = float(roc_auc_score([1 if y == 0 else 0 for y in ys], fake_scores))
    else: result["roc_auc_fake"] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test_dataset")
    parser.add_argument("--model-out", default="models/baseline_model.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_items, val_items, test_items = collect_split(Path(args.data))
    print(f"Files: train={len(train_items)}, validation={len(val_items)}, test={len(test_items)}")
    train_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train_loader = DataLoader(MediaDataset(train_items, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(MediaDataset(val_items, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(MediaDataset(test_items, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(.4), nn.Linear(model.fc.in_features, 2))
    model.to(device)
    # Compute weights only from the training partition; class index 0=fake, 1=real.
    counts = np.bincount([label for _, label in train_items], minlength=2)
    weights = len(train_items) / (2.0 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    best_f1, best_state, best_epoch = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train(); loss_sum = 0.0; seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True); logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step(); loss_sum += loss.item() * len(y); seen += len(y)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={loss_sum/max(seen,1):.4f} | val_accuracy={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_state, best_epoch = val_metrics["macro_f1"], {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, epoch + 1
    if best_state is None: raise RuntimeError("No model checkpoint was selected.")
    model.load_state_dict(best_state); model.to(device)
    model_out = Path(args.model_out); model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    test_metrics = evaluate(model, test_loader, device)
    report_path = model_out.with_name(model_out.stem + "_metrics.json")
    report = {"seed": SEED, "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
              "split_sizes": {"train":len(train_items),"validation":len(val_items),"test":len(test_items)},
              "test_metrics": test_metrics,
              "notes": ["Videos are represented by their first readable frame.", "This is file-level splitting; if files are extracted frames from the same source video, split by source video before training.", "Metrics are specific to this dataset split and are not a guarantee of real-world accuracy."]}
    report_path.write_text(json.dumps(report, indent=2))
    print("Test confusion matrix (true rows FAKE, REAL; predicted columns FAKE, REAL):")
    print(np.array(test_metrics["confusion_matrix"]))
    print(f"Test accuracy: {test_metrics['accuracy']:.4f} | macro-F1: {test_metrics['macro_f1']:.4f} | ROC-AUC(fake): {test_metrics['roc_auc_fake']}")
    print(f"Saved checkpoint: {model_out}\nSaved metrics: {report_path}")

if __name__ == "__main__": main()
