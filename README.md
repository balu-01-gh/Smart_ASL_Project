# Smart ASL Recognition

Smart ASL Recognition is a deep learning project for recognizing American Sign Language (ASL) gestures from video using pose estimation and sequence models. It leverages TensorFlow/Keras, MediaPipe, and OpenCV for feature extraction, model training, and real-time prediction.

## Overview

This project provides an end-to-end pipeline for ASL gesture recognition:

- **Pose Extraction:** Uses MediaPipe to extract 33 keypoint landmarks from each video frame, representing the human body pose. OpenCV is used for video frame processing and visualization.
- **Feature Engineering:** Landmarks are normalized and converted into feature vectors for each frame. Sequences of pose vectors are constructed for each gesture sample.
- **Model Architecture:**
  - Deep LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit) networks are used to learn temporal patterns in sequential pose data.
  - The model consists of stacked LSTM/GRU layers, followed by dense layers and dropout for regularization.
  - Trained using TensorFlow/Keras.
  - A robust, production-ready model is exported as `asl_robust_pose_model_89.keras`.
- **Data Handling:**
  - Data is split using group-aware splitting to prevent data leakage (ensuring no overlap between train/val/test groups).
  - Evaluation includes per-class accuracy and confusion matrix for transparency.
- **Deployment:**
  - Flask serves the trained model via a web interface for real-time gesture prediction.
  - The app can be started with a single command (`start.bat`).


  ## Dataset

This project uses a preprocessed subset of the WLASL100 dataset for American Sign Language recognition. The dataset has been processed as follows:
- **Frames:** Video samples are split into individual frames.
- **Pose Extraction:** MediaPipe is used to extract 33 pose landmarks from each frame.
- **Preprocessed Data:**
  - Extracted pose sequences and labels are stored as numpy arrays in the `processed_data/` directory.
  - The `dataset/` directory contains the original and preprocessed data, organized into `train/`, `val/`, and `test/` splits, each with `frames/` and `pose/` subfolders.
- **Training:**
  - Models are trained on these pose sequences, enabling efficient and accurate gesture recognition.

This preprocessing ensures the model is trained on high-quality, structured data, and enables reproducible experiments.

## Features
- Pose-based ASL gesture recognition
- LSTM/GRU sequence models
- Real-time video prediction
- Flask web interface
- Group-aware data splitting to prevent data leakage

## Project Structure
```
Smart_ASL_Project/
├── app.py                  # Flask web app for real-time prediction
├── predict_video.py        # Script for video-based prediction
├── train_model_pose_lstm.py# Model training script
├── asl_robust_pose_model_89.keras  # Trained model (keep for inference)
├── asl_pose_lstm_best.h5   # Best model checkpoint (from training)
├── per_class_accuracy.csv  # Per-class accuracy report
├── confusion_matrix.png    # Confusion matrix visualization
├── start.bat               # Windows batch file to launch the app
├── dataset/                # Raw and preprocessed data
├── processed_data/         # Numpy arrays and labels
├── models/                 # Saved models
├── static/, templates/     # Web UI assets
├── mp_env/                 # Python virtual environment (not for Git)
├── venv/                   # (optional) Python virtual environment
```

## Setup
1. **Clone the repository**
2. **Create and activate a virtual environment** (recommended):
   - Windows: `python -m venv mp_env && mp_env\Scripts\activate`
3. **Install dependencies:**
   - `pip install -r requirements.txt`
4. **Run the app:**
   - `start.bat` (Windows)
   - Or manually: `python app.py`

## Usage
- Access the web interface at `http://localhost:5000` after running the app.
- Use `predict_video.py` for batch video predictions.
- Use `train_model_pose_lstm.py` to retrain the model with new data.

## Example
To run the app and see a live demo:
1. Open a terminal in the project directory.
2. Run `start.bat` (Windows) or `python app.py` (after activating your environment).
3. Open your browser and go to [http://localhost:5000](http://localhost:5000).

You should see the web interface for uploading or streaming ASL videos and getting predictions.

## Notes
- Exclude `mp_env/`, `venv/`, and large model/data files from Git using `.gitignore`.
- For best results, use high-quality, well-lit video for predictions.

## License
MIT License

## Authors
Jayavarapu Bala Subrahmanyam

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.



