"""Shared inference for image and frame-based video classification.

Important: train_advanced.py trains on full images / the first full video frame,
not face crops. This predictor therefore uses full frames to keep preprocessing
consistent with the existing checkpoint. Video output is frame-level score
aggregation, not temporal modelling.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.models.resnet_detector import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class DeepfakePredictor:
    def __init__(self, model_path: str | os.PathLike = "models/baseline_model.pth", device: str | None = None):
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}. Train the ResNet-50 first "
                "or set DEEPFAKE_MODEL_PATH to the checkpoint location."
            )
        self.model = build_model(pretrained=False).to(self.device)
        try:
            try:
                state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except TypeError:  # PyTorch versions before weights_only support
                state = torch.load(self.model_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Handle checkpoints saved from DataParallel.
            if isinstance(state, dict) and state and all(k.startswith("module.") for k in state):
                state = {k.removeprefix("module."): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {self.model_path} into the ResNet-50 architecture. "
                "The checkpoint may belong to the old custom CNN; use the checkpoint "
                "created by train_advanced.py."
            ) from exc
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def _predict_rgb(self, rgb: np.ndarray) -> dict[str, Any]:
        if rgb is None or rgb.size == 0:
            raise ValueError("The image/frame is empty or invalid.")
        image = Image.fromarray(rgb).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        fake_p = float(probs[0].item())
        real_p = float(probs[1].item())
        label = "FAKE" if fake_p >= real_p else "REAL"
        return {
            "prediction": label,
            "fake_probability": round(fake_p, 6),
            "real_probability": round(real_p, 6),
            "confidence": round(max(fake_p, real_p), 6),
        }

    def predict_image(self, path: str | os.PathLike) -> dict[str, Any]:
        path = Path(path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Unsupported image format.")
        try:
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("The uploaded file is not a readable image.") from exc
        result = self._predict_rgb(rgb)
        result.update({"media_type": "image", "frames_analyzed": 1, "faces_detected": None})
        return result

    def predict_video(self, path: str | os.PathLike, max_frames: int = 12) -> dict[str, Any]:
        path = Path(path)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Unsupported video format.")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("Could not open the uploaded video.")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ValueError("The video contains no readable frames.")
            indices = np.unique(np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int))
            results = []
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results.append(self._predict_rgb(rgb))
            if not results:
                raise ValueError("Could not decode any sampled video frames.")
        finally:
            cap.release()
        # Mean frame probability is a simple baseline aggregation, not temporal analysis.
        fake_p = float(np.mean([r["fake_probability"] for r in results]))
        real_p = float(np.mean([r["real_probability"] for r in results]))
        label = "FAKE" if fake_p >= real_p else "REAL"
        return {
            "prediction": label,
            "fake_probability": round(fake_p, 6),
            "real_probability": round(real_p, 6),
            "confidence": round(max(fake_p, real_p), 6),
            "media_type": "video",
            "frames_analyzed": len(results),
            "faces_detected": None,
            "note": "Frame-based classification with mean probability aggregation; no temporal model is used.",
        }
