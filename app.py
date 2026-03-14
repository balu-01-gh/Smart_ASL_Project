"""
Smart ASL Recognition - Flask Web Application
Serves webcam-based ASL sign language prediction through a web interface.
"""

import os
import json
import base64
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
import tensorflow as tf
import joblib
from collections import deque
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from keras.models import load_model
def safe_load_model(filepath):
    """Fix Keras3 compatibility for old models - legacy InputLayer"""
    try:
        import tensorflow.keras as keras
        from tensorflow.keras.layers import InputLayer
        custom_objects = {'InputLayer': InputLayer}
        model = keras.models.load_model(filepath, custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("[INFO] Model loaded with legacy InputLayer fix")
        return model
    except Exception as e:
        print(f"[ERROR] Legacy fix failed: {e}")
        raise
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# =====================
# App Configuration
# =====================
app = Flask(__name__)
CORS(app)

# =====================
# Settings
# =====================
MODEL_PATH = "asl_robust_pose_model_89.keras"
LABELS_PATH = "model_labels.json"  # Local copy if exists
SCALER_PATH = "processed_data/feature_scaler.pkl"
SEQ_LEN = 1
EARLY_PREDICT_MIN = 1        # Predict IMMEDIATELY (from 1st frame)
IMG_SIZE = 224               # Match training size for best accuracy
CONFIDENCE_THRESHOLD = 0.30  # Slightly lower to allow faster initial feedback
LOG_EVERY_N = 10              # Only log detection info every N frames

print("[INFO] Loading ASL model...")
model = safe_load_model(MODEL_PATH)
_model_fn = tf.function(model, reduce_retracing=True)

print("[INFO] Loading labels...")
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
# Invert mapping: index (int) -> word (str)
idx_to_label = {v: k for k, v in label_map.items()}
print(f"[INFO] Total classes: {len(idx_to_label)}")

print("[INFO] Loading feature scaler...")
scaler = joblib.load(SCALER_PATH)

print("[INFO] Loading MobileNetV2 feature extractor...")
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)   # explicit shape avoids weight-mismatch warning
)
# Compile with tf.function for fast single-sample inference (3-5x faster than .predict())
_extract_fn = tf.function(feature_extractor, reduce_retracing=True)
# Warm up GPU/CPU kernel with a dummy forward pass so first real frame isn't slow
_dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
_ = _extract_fn(_dummy, training=False).numpy()
print("[INFO] Feature extractor warmed up")

# Warm up LSTM model with a dummy sequence (eliminates first-prediction cold start)
_dummy_seq = np.zeros((1, SEQ_LEN, 1280), dtype="float32")
_ = _model_fn(_dummy_seq, training=False).numpy()
print("[INFO] LSTM model warmed up")

# =====================
# MediaPipe HandLandmarker (Tasks API, thread-safe)
# =====================
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
HAND_LANDMARKER_PATH = "hand_landmarker.task"
if not os.path.exists(HAND_LANDMARKER_PATH):
    raise FileNotFoundError(f"[ERROR] {HAND_LANDMARKER_PATH} not found. Download it from https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task and place in project root.")

hand_landmarker_options = HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=VisionTaskRunningMode.IMAGE
)
hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)
hands_lock = threading.Lock()
print("[INFO] MediaPipe HandLandmarker (Tasks API) initialized")

# =====================
# Prediction Buffers
# pred_buffer=7 gives stronger majority-vote stability vs 5
# =====================
sequence = deque(maxlen=SEQ_LEN)
pred_buffer = deque(maxlen=7)

# =====================
# Statistics Tracking
# =====================
stats = {
    "total_frames": 0,
    "hands_detected": 0,
    "predictions_made": 0,
    "prediction_history": deque(maxlen=50),
    "start_time": None,
    "last_prediction": None,
    "last_confidence": 0,
}
frame_counter = 0


# =====================
# Routes
# =====================
@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Serve the about page."""
    return render_template("about.html")


@app.route("/api/labels", methods=["GET"])
def get_labels():
    """Return all ASL labels the model can recognize."""
    return jsonify({
        "labels": list(idx_to_label.values()),
        "count": len(idx_to_label)
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receive a base64-encoded frame from the webcam,
    process it, and return the ASL prediction.
    """
    global sequence, pred_buffer, frame_counter

    try:
        t_start = time.time()

        if stats["start_time"] is None:
            stats["start_time"] = t_start

        data = request.get_json()
        if not data or "frame" not in data:
            return jsonify({"error": "No frame data received"}), 400

        # Decode base64 image
        frame_data = data["frame"]
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        stats["total_frames"] += 1
        frame_counter += 1


        # Hand Detection - MediaPipe Tasks API (fix: wrap numpy array as MediaPipe Image)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        with hands_lock:
            result: HandLandmarkerResult = hand_landmarker.detect(mp_image)

        hand_detected = len(result.hand_landmarks) > 0

        # Throttled logging — only every N frames
        if frame_counter % LOG_EVERY_N == 0:
            print(f"[DETECT] hand={hand_detected}, frame={frame.shape[1]}x{frame.shape[0]}, total={stats['total_frames']}")

        # Extract landmark data for visual overlay
        landmarks_data = []
        if hand_detected:
            for hand_landmarks in result.hand_landmarks:
                lm_list = []
                for lm in hand_landmarks:
                    lm_list.append({"x": round(lm.x, 4), "y": round(lm.y, 4)})
                landmarks_data.append(lm_list)

        if hand_detected:
            stats["hands_detected"] += 1

        if not hand_detected:
            # Don't clear the sequence buffer — just skip this frame
            return jsonify({
                "prediction": None,
                "confidence": 0,
                "message": "No hand detected",
                "hand_detected": False,
                "landmarks": [],
                "buffer_size": len(sequence),
                "required_frames": SEQ_LEN,
                "process_ms": round((time.time() - t_start) * 1000, 1)
            })

        # Feature Extraction — direct call is 3-5x faster than .predict() for single images
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        if len(img.shape) == 3 and img.shape[2] == 3: # Check if BGR
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype("float32"))
        img = np.expand_dims(img, axis=0)

        features = _extract_fn(img, training=False).numpy()[0]
        
        # Apply normalization (CRITICAL: must match training preprocessing)
        features = scaler.transform(features.reshape(1, -1))[0]
        
        sequence.append(features)

        # Prediction immediately from the first frame
        buf_len = len(sequence)
        seq_list = list(sequence)
        # No padding needed since SEQ_LEN = 1
        seq_input = np.expand_dims(seq_list, axis=0).astype("float32")
        preds = _model_fn(seq_input, training=False).numpy()[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])

        pred_buffer.append(top_idx)

        # Majority voting for stability (still used for smoothing)
        final_idx = max(set(pred_buffer), key=list(pred_buffer).count)
        word = idx_to_label[final_idx]

        # Get top 3 predictions
        top3_indices = np.argsort(preds)[-3:][::-1]
        top3 = [
            {"word": idx_to_label[int(i)], "confidence": round(float(preds[i]), 3)}
            for i in top3_indices
        ]

        process_ms = round((time.time() - t_start) * 1000, 1)

        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "prediction": None,
                "confidence": round(confidence, 3),
                "message": f"Low confidence ({round(confidence * 100)}%)",
                "hand_detected": True,
                "landmarks": landmarks_data,
                "top3": top3,
                "buffer_size": len(sequence),
                "required_frames": SEQ_LEN,
                "process_ms": process_ms
            })

        # Track statistics
        stats["predictions_made"] += 1
        stats["last_prediction"] = word
        stats["last_confidence"] = round(confidence, 3)
        stats["prediction_history"].append({
            "word": word,
            "confidence": round(confidence, 3),
            "timestamp": time.time()
        })

        return jsonify({
            "prediction": word,
            "confidence": round(confidence, 3),
            "hand_detected": True,
            "landmarks": landmarks_data,
            "top3": top3,
            "buffer_size": len(sequence),
            "required_frames": SEQ_LEN,
            "process_ms": process_ms
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset prediction buffers."""
    global sequence, pred_buffer
    sequence.clear()
    pred_buffer.clear()
    return jsonify({"status": "reset", "message": "Buffers cleared"})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    uptime = round(time.time() - stats["start_time"], 1) if stats["start_time"] else 0
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": len(idx_to_label),
        "uptime_seconds": uptime
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return prediction statistics."""
    uptime = round(time.time() - stats["start_time"], 1) if stats["start_time"] else 0
    fps = round(stats["total_frames"] / uptime, 1) if uptime > 0 else 0
    detection_rate = round(stats["hands_detected"] / max(stats["total_frames"], 1) * 100, 1)

    return jsonify({
        "total_frames": stats["total_frames"],
        "hands_detected": stats["hands_detected"],
        "predictions_made": stats["predictions_made"],
        "detection_rate_pct": detection_rate,
        "avg_fps": fps,
        "uptime_seconds": uptime,
        "last_prediction": stats["last_prediction"],
        "last_confidence": stats["last_confidence"],
        "recent_history": list(stats["prediction_history"])[-10:]
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return recent prediction history."""
    return jsonify({
        "history": list(stats["prediction_history"]),
        "count": len(stats["prediction_history"])
    })


@app.route("/api/debug", methods=["GET"])
def debug_test():
    """Test MediaPipe HandLandmarker with a synthetic image."""
    import time
    # Create a blank test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8) + 200
    with hands_lock:
        start = time.time()
        result: HandLandmarkerResult = hand_landmarker.detect(test_img)
        elapsed = time.time() - start
    return jsonify({
        "mediapipe_working": True,
        "detection_time_ms": round(elapsed * 1000, 1),
        "hand_found_on_blank": len(result.hand_landmarks) > 0,
        "message": "MediaPipe Tasks API is operational. Try showing your hand to the camera."
    })


# =====================
# Run
# =====================
if __name__ == "__main__":
    print("[INFO] Starting Smart ASL Recognition Server...")
    print("[INFO] Open http://localhost:5000 in your browser")
    app.run(debug=False, host="0.0.0.0", port=5000)
