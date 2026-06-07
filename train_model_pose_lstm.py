import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Bidirectional
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

DATA_DIR = "processed_data"

# ── Load Data ─────────────────────────────────────────────────────────────────
print("[INFO] Loading datasets...")

# Helper to load data safely
def load_split(name):
    try:
        X = np.load(os.path.join(DATA_DIR, "X_pose_seq.npy"))
        y = np.load(os.path.join(DATA_DIR, "y_pose_seq.npy"))
        print(f"Using combined dataset: X_pose_seq.npy {X.shape}")
        return X, y
    except FileNotFoundError:
        print(f"[ERROR] Could not load dataset. Found files: feature_scaler.pkl, labels.json, X_pose_seq.npy")
        exit(1)

X, y = load_split("pose")
# Split 80/20
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Train samples: {X_train.shape}")
print(f"Val   samples: {X_val.shape}")

# ── Robust Normalization ──────────────────────────────────────────────────────
# Fit scaler ONLY on training data
print("[INFO] Normalizing...")
scaler = StandardScaler()

samples, timesteps, features = X_train.shape
# Flatten (N, T, F) -> (N*T, F)
X_train_reshaped = X_train.reshape(-1, features)
# Fit on train
X_train_reshaped = scaler.fit_transform(X_train_reshaped)
# Reshape back
X_train = X_train_reshaped.reshape(samples, timesteps, features)

# Transform Val (using train statistics)
val_samples, _, _ = X_val.shape
X_val_reshaped = X_val.reshape(-1, features)
X_val_reshaped = scaler.transform(X_val_reshaped)
X_val = X_val_reshaped.reshape(val_samples, timesteps, features)

# Save scaler for inference
joblib.dump(scaler, os.path.join(DATA_DIR, "feature_scaler.pkl"))
print("Scaler saved.")

# ── Class Weights ─────────────────────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
all_unique = np.unique(np.concatenate((y_train, y_val)))
num_classes = int(all_unique.max()) + 1
print(f"Labels range [0, {all_unique.max()}] classes: {len(all_unique)} max_classes: {num_classes}")

# ── Model Architecture ────────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(timesteps, features)),

    # Bidirectional GRU helps learn past AND future context
    Bidirectional(GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
    BatchNormalization(),
    
    Bidirectional(GRU(64, dropout=0.3, recurrent_dropout=0.2)),
    BatchNormalization(),

    Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Lower LR for stability
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)
checkpoint = ModelCheckpoint(
    filepath="asl_pose_lstm_best.h5",
    monitor="val_accuracy", save_best_only=True, verbose=1
)

# ── Train ─────────────────────────────────────────────────────────────────────
# Note: No shuffle=True needed for val_data as it's separate
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,    # 200 epochs as data is larger and augmented
    batch_size=32,
    class_weight=class_weights,
    callbacks=[lr_scheduler, early_stop, checkpoint]
)

model.save("asl_pose_lstm.h5")

best_val   = max(history.history["val_accuracy"])
best_train = max(history.history["accuracy"])
print(f"\nTraining completed!")
print(f"Best Train Accuracy : {best_train*100:.2f}%")
print(f"Best Val   Accuracy : {best_val  *100:.2f}%")


