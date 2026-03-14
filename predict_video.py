import mediapipe as mp
import tensorflow as tf
import cv2
import numpy as np
import os
import json
import joblib
from collections import deque

from keras.models import load_model
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# =====================
# Settings
# =====================
MODEL_PATH = "asl_pose_lstm_best.keras"
LABELS_PATH = "processed_data/labels.json"
SCALER_PATH = "processed_data/feature_scaler.pkl"
SEQ_LEN = 13
IMG_SIZE = 224  # Match training size for best accuracy


# =====================
# Load Model
# =====================
print("Loading model...")
model = load_model(MODEL_PATH)
# Wrap in tf.function for fast inference (no .predict() overhead)
_model_fn = tf.function(model, reduce_retracing=True)
# Warm up LSTM kernel
_dummy_seq = np.zeros((1, SEQ_LEN, 1280), dtype="float32")
_ = _model_fn(_dummy_seq, training=False).numpy()
print("LSTM model warmed up")


# =====================
# Load Scaler
# =====================
print("Loading feature scaler...")
scaler = joblib.load(SCALER_PATH)

# =====================
# Load Labels
# =====================
print("Loading labels...")

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

# labels.json format: {0: "apple", 1: "boy", ...}
idx_to_label = {int(k): v for k, v in label_map.items()}

print("Total classes:", len(idx_to_label))


# =====================
# MobileNet Feature Extractor
# =====================
print("Loading MobileNet...")
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)   # explicit shape avoids weight-mismatch warning
)
# Wrap in tf.function + warm up to eliminate first-frame slowdown
_extract_fn = tf.function(feature_extractor, reduce_retracing=True)
_dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
_ = _extract_fn(_dummy, training=False)
print("Feature extractor warmed up")


# =====================
# MediaPipe Hands (tracking mode = much faster for live webcam)
# =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =====================
# Buffers
# pred_buffer=7 gives stronger majority-vote stability
# =====================
sequence = deque(maxlen=SEQ_LEN)
pred_buffer = deque(maxlen=7)   # increased from 5 for stable prediction


# =====================
# Webcam
# =====================
cap = cv2.VideoCapture(0)
print("Camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ---------------------
    # Hand Detection
    # ---------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_text = "Show hand"

    if results.multi_hand_landmarks:

        # Draw hand
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # ---------------------
        # Feature Extraction — direct call, 3-5x faster than .predict() for single images
        # ---------------------
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB for model
        img = preprocess_input(img.astype("float32"))
        img = np.expand_dims(img, axis=0)

        features = _extract_fn(img, training=False).numpy()[0]
        sequence.append(features)

        if len(sequence) == SEQ_LEN:
            # ---------------------
            # Prediction
            # ---------------------
            seq_arr = np.array(sequence)     # (13, 1280)
            seq_scaled = scaler.transform(seq_arr)  # Normalize
            
            seq_input = np.expand_dims(seq_scaled, axis=0).astype("float32") # (1, 13, 1280)

            preds = _model_fn(seq_input, training=False).numpy()[0]
            top_idx = np.argmax(preds)
            confidence = preds[top_idx]

            pred_buffer.append(top_idx)

            # Majority voting for stability
            final_idx = max(set(pred_buffer), key=pred_buffer.count)
            word = idx_to_label[final_idx]

            display_text = f"{word} ({confidence:.2f})"

    else:
        sequence.clear()
        pred_buffer.clear()

    # ---------------------
    # Display
    # ---------------------
    cv2.putText(
        frame,
        display_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
