from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)
CORS(app)

# ── Model & class config ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "skin_disease_mobilenet_model.h5")

# Updated class names based on your dataset structure
CLASS_NAMES = [
    "BA-cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles",
]

IMG_SIZE = (224, 224)

print("Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded ✓")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def preprocess(img_bytes: bytes) -> np.ndarray:
    """Decode image bytes → normalised (1, 224, 224, 3) float32 array."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0), img


def make_gradcam_heatmap(img_array: np.ndarray, pred_index: int) -> np.ndarray:
    """Generate a Grad-CAM heatmap for the predicted class."""
    # Find last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    if last_conv is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)
    return heatmap


def overlay_heatmap(orig_img: np.ndarray, heatmap: np.ndarray) -> str:
    """Resize heatmap, overlay on original image, return base64 JPEG string."""
    h, w = orig_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    orig_bgr = cv2.cvtColor((orig_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
    _, buffer = cv2.imencode(".jpg", superimposed)
    return base64.b64encode(buffer).decode("utf-8")


# ── Routes ───────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Skin Disease Detection API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img_bytes = request.files["file"].read()

    try:
        img_array, orig_img = preprocess(img_bytes)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    preds = model.predict(img_array)[0]
    pred_index = int(np.argmax(preds))
    confidence = float(preds[pred_index])
    prediction = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else f"Class {pred_index}"

    heatmap_b64 = ""
    try:
        heatmap = make_gradcam_heatmap(img_array, pred_index)
        if heatmap is not None:
            heatmap_b64 = overlay_heatmap(orig_img, heatmap)
    except Exception:
        pass  # heatmap is optional

    return jsonify(
        {
            "prediction": prediction,
            "confidence": confidence,
            "heatmap": heatmap_b64,
        }
    )


# ── Entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
