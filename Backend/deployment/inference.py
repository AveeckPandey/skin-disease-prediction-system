import os
import json
import numpy as np
import tensorflow as tf
import cv2
import base64

# Define class names in alphabetical order (same as local app.py)
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

def model_fn(model_dir):
    """Load the model from the model_dir."""
    model_path = os.path.join(model_dir, "skin_disease_mobilenet_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse the input request body."""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        img_b64 = input_data['image']
        img_bytes = base64.b64decode(img_b64)
    elif request_content_type == 'image/jpeg' or request_content_type == 'image/png':
        img_bytes = request_body
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_fn(input_data, model):
    """Run prediction on the processed input data."""
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, accept):
    """Format the prediction output."""
    preds = prediction[0]
    pred_index = int(np.argmax(preds))
    confidence = float(preds[pred_index])
    result = {
        "prediction": CLASS_NAMES[pred_index],
        "confidence": confidence,
        "all_predictions": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    }
    return json.dumps(result), "application/json"
