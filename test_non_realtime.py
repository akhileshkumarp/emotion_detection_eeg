# test_bilstm_single.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved model and preprocessing tools
model = load_model("bilstm_eeg_model.h5")
scaler = joblib.load("scaler.save")
encoder = joblib.load("label_encoder.save")

# -----------------------------
# TEST FUNCTION (Single Sample)
# -----------------------------
def predict_emotion_single(eeg_value):
    """
    eeg_value: single EEG reading in microvolts (float)
    """
    timesteps = 10  # same as training
    # Create artificial sequence by repeating the single value
    seq = np.array([eeg_value] * timesteps).reshape(-1, 1)
    seq_scaled = scaler.transform(seq)
    X_input = np.expand_dims(seq_scaled, axis=0)

    # Predict
    pred = model.predict(X_input)
    label = encoder.inverse_transform([np.argmax(pred)])[0]
    confidence = float(np.max(pred))

    print(f"EEG Input: {eeg_value:.2f} µV")
    print(f"Predicted Emotion: {label} (Confidence: {confidence:.2f})")
    return label

# -----------------------------
# EXAMPLE TEST
# -----------------------------
# Example single EEG value
sample_value = 37.16
predict_emotion_single(sample_value)
