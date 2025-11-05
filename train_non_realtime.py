import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("eeg_emotion_dataset.csv")

# EEG signal
X = data["EEG_microvolts"].values.reshape(-1, 1)
y = data["Label"].values

# Normalize EEG data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Reshape for LSTM [samples, timesteps, features]
# We use a sliding window of 10 samples for temporal context
timesteps = 10
def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:(i+timesteps)])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, timesteps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, timesteps)

# -----------------------------
# BUILD BiLSTM MODEL
# -----------------------------
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=False, input_shape=(timesteps, 1))),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint("bilstm_eeg_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=30,
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq),
    callbacks=[checkpoint],
    verbose=1
)

print("✅ Training complete. Best model saved as 'bilstm_eeg_model.h5'.")

# Optionally save encoders and scalers
import joblib
joblib.dump(scaler, "scaler.save")
joblib.dump(encoder, "label_encoder.save")
