"""Flask web application for ResNet-50 image and video inference."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src.inference.predictor import DeepfakePredictor, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("DEEPFAKE_MODEL_PATH", BASE_DIR / "models" / "baseline_model.pth"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
_predictor = None
_predictor_error = None


def get_predictor():
    global _predictor, _predictor_error
    if _predictor is None and _predictor_error is None:
        try:
            _predictor = DeepfakePredictor(MODEL_PATH)
        except Exception as exc:
            _predictor_error = str(exc)
    if _predictor is None:
        raise RuntimeError(_predictor_error or "Model could not be loaded.")
    return _predictor


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _predictor is not None,
                    "model_path": str(MODEL_PATH)})


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"status": "error", "message": f"File too large. Maximum upload is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}), 413


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file was uploaded."}), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"status": "error", "message": "Choose a file first."}), 400
    safe_name = secure_filename(upload.filename)
    if not safe_name or Path(safe_name).suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return jsonify({"status": "error", "message": "Unsupported file type. Upload JPG, PNG, WEBP, MP4, AVI, MOV, MKV or WEBM."}), 400
    suffix = Path(safe_name).suffix.lower()
    temp_path = None
    try:
        # NamedTemporaryFile avoids trusting a user-supplied path and guarantees unique names.
        with tempfile.NamedTemporaryFile(prefix="deepfake_", suffix=suffix, delete=False) as tmp:
            temp_path = Path(tmp.name)
        upload.save(temp_path)
        predictor = get_predictor()
        result = predictor.predict_video(temp_path) if suffix in VIDEO_EXTENSIONS else predictor.predict_image(temp_path)
        result.update({"status": "success", "filename": safe_name})
        return jsonify(result)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 422
    except Exception:
        app.logger.exception("Unexpected error while processing uploaded media")
        return jsonify({"status": "error", "message": "Could not process this file. Check the server logs."}), 500
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Could not remove temporary upload: %s", temp_path)


if __name__ == "__main__":
    # Development server only. Use a production WSGI server for deployment.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
