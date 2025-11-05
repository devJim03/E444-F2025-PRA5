import os
import logging
import threading
from typing import Optional
from flask import Flask, request, jsonify, render_template_string

# Flask app (Elastic Beanstalk Procfile expects "application:application")
application = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve artifact paths relative to this file; allow env overrides (empty env won't override)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(BASE_DIR, "basic_classifier.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or os.path.join(BASE_DIR, "count_vectorizer.pkl")

# Log resolved paths
logger.info("CWD: %s", os.getcwd())
logger.info("Resolved MODEL_PATH: %s", MODEL_PATH)
logger.info("Resolved VECTORIZER_PATH: %s", VECTORIZER_PATH)

# Global variables for loaded artifacts
_loaded_model: Optional[object] = None
_vectorizer: Optional[object] = None
_artifact_lock = threading.Lock()

# Artifact loading
def _load_artifacts_once() -> None:
    """Lazily load model and vectorizer once per process."""
    global _loaded_model, _vectorizer
    if _loaded_model is not None and _vectorizer is not None:
        return
    with _artifact_lock:
        if _loaded_model is None or _vectorizer is None:
            import pickle
            logger.info("Loading artifacts...")
            with open(MODEL_PATH, "rb") as mf:
                _loaded_model = pickle.load(mf)
            with open(VECTORIZER_PATH, "rb") as vf:
                _vectorizer = pickle.load(vf)
            logger.info("Artifacts loaded.")

# Inference function
def _predict_text(message: str) -> str:
    """Run inference and return the predicted class as a string label."""
    _load_artifacts_once()
    X = _vectorizer.transform([message])
    pred = _loaded_model.predict(X)
    # pred[0] could be a numpy scalar; normalize to native str
    val = pred[0]
    val_py = val.item() if hasattr(val, "item") else val
    return str(val_py)

# Eager load artifacts in a background thread at startup
def _eager_load_background():
    try:
        _load_artifacts_once()
    except Exception as e:
        # Log and continue; app remains healthy and will lazy-load on first request
        logger.warning("Background eager load failed: %s", e, exc_info=True)


# Non-blocking eager load at startup
threading.Thread(target=_eager_load_background, daemon=True).start()

DEMO_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Model Prediction Demo</title>
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        margin: 0;
        padding: 2rem;
        background-color: #f8f9fa;
        color: #333;
      }
      .container {
        max-width: 600px;
        margin: auto;
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
      }
      h1, h2 {
        color: #212529;
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 0.5rem;
      }
      form {
        display: flex;
        flex-direction: column;
        gap: 1rem;
      }
      textarea {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 1rem;
        resize: vertical;
        box-sizing: border-box; /* Important for padding */
      }
      button {
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        color: #fff;
        background-color: #007bff;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.2s;
      }
      button:hover {
        background-color: #0056b3;
      }
      .result-box, .status-box {
        margin-top: 1.5rem;
        padding: 1rem;
        border-radius: 4px;
        word-wrap: break-word;
      }
      .status-box.success {
        background-color: #e6f7ec;
        border: 1px solid #b7e1cd;
        color: #0a3622;
      }
      .status-box.error {
        background-color: #fbeae5;
        border: 1px solid #f5c0b1;
        color: #5f2120;
      }
      .result-box.prediction {
        background-color: #e6f7ff;
        border: 1px solid #b3e0ff;
      }
      .result-box.error {
        background-color: #fbeae5;
        border: 1px solid #f5c0b1;
        color: #5f2120;
      }
      .result-box h3 {
        margin-top: 0;
      }
      label {
        font-weight: 600;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Text Classifier Demo</h1>
      
      {% if model_loaded %}
        <div class="status-box success">
          <strong>Status:</strong> Model is loaded and ready.
        </div>
      {% else %}
        <div class="status-box error">
          <strong>Status:</strong> Model artifacts not found or not loaded.
        </div>
      {% endif %}

      <h2>Enter Text</h2>
      <form action="/predict-form" method="post">
        <label for="message">Text to classify:</label>
        <textarea id="message" name="message" rows="5" placeholder="Type your text here..."></textarea>
        <button type="submit" {% if not model_loaded %}disabled{% endif %}>
          Predict
        </button>
      </form>

      {% if error %}
        <div class="result-box error">
          <h3>Error</h3>
          <p>{{ error }}</p>
        </div>
      {% endif %}
      
      {% if prediction %}
        <div class="result-box prediction">
          <h3>Prediction</h3>
          <p><strong>{{ prediction }}</strong></p>
        </div>
      {% endif %}

    </div>
  </body>
</html>
"""

# Routes
@application.get("/")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(_loaded_model is not None and _vectorizer is not None),
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH,
    }), 200

# Demo page rendering endpoint
@application.get("/demo")
def demo():
    return render_template_string(
        DEMO_HTML,
        model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
        model_path=MODEL_PATH,
        prediction=None,
        error=None,
    )

# Form submission endpoint for demo page
@application.post("/predict-form")
def predict_form():
    message = (request.form.get("message") or "").strip()
    if not message:
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Field 'message' is required and must be non-empty.",
        ), 400
    try:
        label = _predict_text(message)
        return render_template_string(
            DEMO_HTML,
            model_loaded=True,
            model_path=MODEL_PATH,
            prediction=label,
            error=None,
        )
    except FileNotFoundError:
        return render_template_string(
            DEMO_HTML,
            model_loaded=False,
            model_path=MODEL_PATH,
            prediction=None,
            error="Model artifacts not found on server.",
        ), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Inference failed.",
        ), 500

# JSON API endpoint for predictions
@application.post("/predict")
def predict_json():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Field 'message' is required and must be non-empty."}), 400
    try:
        label = _predict_text(message)
        return jsonify({"label": label}), 200
    except FileNotFoundError:
        return jsonify({"error": "Model artifacts not found on server."}), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return jsonify({"error": "Inference failed."}), 500


if __name__ == "__main__":
    # Local dev run; in EB, Gunicorn (from Procfile) will host the app
    port = int(os.getenv("PORT", "8000"))
    application.run(host="0.0.0.0", port=port, debug=False)
