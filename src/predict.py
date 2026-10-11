"""Command-line image/video inference using the same model as the Flask app."""
import argparse
import json
from pathlib import Path
from src.inference.predictor import DeepfakePredictor, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description="Classify an image or sample frames from a video.")
    parser.add_argument("media", help="Path to an image or video")
    parser.add_argument("--model", default="models/baseline_model.pth", help="ResNet-50 checkpoint path")
    parser.add_argument("--frames", type=int, default=12, help="Maximum evenly sampled video frames")
    args = parser.parse_args()
    path = Path(args.media)
    predictor = DeepfakePredictor(args.model)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        result = predictor.predict_image(path)
    elif path.suffix.lower() in VIDEO_EXTENSIONS:
        result = predictor.predict_video(path, max_frames=max(1, args.frames))
    else:
        parser.error("Unsupported media extension")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
