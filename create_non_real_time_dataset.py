import csv
import random
import time

# ----------------------------
# CONFIGURATION
# ----------------------------
num_samples_per_class = 4000   # how many samples for each emotion
output_file = "eeg_emotion_dataset.csv"
labels = ["happy", "sad", "angry"]

# ----------------------------
# FUNCTION TO SIMULATE EEG SIGNAL
# ----------------------------
def generate_eeg_signal(emotion):
    """
    Generate synthetic EEG data in microvolts (µV) based on emotion.
    The values are randomized but follow slightly different patterns.
    """
    if emotion == "happy":
        # Higher average activity, more variability
        base = random.uniform(20, 40)
        noise = random.uniform(-5, 5)
    elif emotion == "sad":
        # Lower amplitude, smoother pattern
        base = random.uniform(5, 15)
        noise = random.uniform(-2, 2)
    elif emotion == "angry":
        # Moderate amplitude but with spikes
        base = random.uniform(15, 30)
        noise = random.uniform(-8, 8)
    else:
        base, noise = 0, 0

    eeg_value = base + noise
    return round(eeg_value, 2)

# ----------------------------
# DATASET GENERATION
# ----------------------------
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Header
    writer.writerow(["EEG_microvolts", "Label"])

    # Generate data for each emotion
    for label in labels:
        for _ in range(num_samples_per_class):
            eeg_value = generate_eeg_signal(label)
            writer.writerow([eeg_value, label])
            # simulate realistic time interval (not required)
            # time.sleep(0.001)

print(f"✅ EEG dataset saved to {output_file} with {num_samples_per_class * len(labels)} samples.")
