
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os
import shutil

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("----- Improved PCA Baseline -----")

start_time = time.time()

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv(r"C:\Users\SRUTHY K S\OneDrive\Desktop\MAINPROJECT\final_research_sequences.csv")

y = df["Label"].values

print("Dataset loaded:", len(df), "blocks")


# --------------------------------------------------
# Convert sequences to text format
# --------------------------------------------------
sequence_text = []

for s in tqdm(df["Sequence"], desc="Processing sequences"):
    
    seq = str(s).strip("[]").split()
    
    sequence_text.append(" ".join(seq))


# --------------------------------------------------
# TF-IDF Feature Extraction
# --------------------------------------------------
print("Building TF-IDF features...")

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sequence_text)

X = X.toarray()


# --------------------------------------------------
# Normalize features
# --------------------------------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# PCA Model
# --------------------------------------------------
print("Training PCA model...")

pca = PCA(
    n_components=30,
    svd_solver="randomized",
    random_state=42
)

X_reduced = pca.fit_transform(X_scaled)

X_reconstructed = pca.inverse_transform(X_reduced)


# --------------------------------------------------
# Reconstruction Error
# --------------------------------------------------
error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)


# Better threshold selection
threshold = np.percentile(error, 90)

print("Threshold selected:", threshold)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
pred = (error > threshold).astype(int)


# --------------------------------------------------
# Metrics
# --------------------------------------------------
total_blocks = len(pred)

anomalies_detected = int(np.sum(pred))

anomaly_rate = anomalies_detected / total_blocks * 100

accuracy = accuracy_score(y, pred)

precision = precision_score(y, pred, zero_division=0)

recall = recall_score(y, pred, zero_division=0)

f1 = f1_score(y, pred, zero_division=0)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# --------------------------------------------------
# Save metrics
# --------------------------------------------------
metrics = {
    "method": "Improved PCA",
    "total_blocks": int(total_blocks),
    "anomalies_detected": int(anomalies_detected),
    "anomaly_rate": float(anomaly_rate),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "threshold": float(threshold),
    "time_seconds": float(time.time() - start_time)
}

with open("traditional_metrics.json", "w") as f:
    
    json.dump(metrics, f, indent=4)

print("Metrics saved to traditional_metrics.json")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
plt.figure(figsize=(7,5))

sns.heatmap(
    confusion_matrix(y, pred),
    annot=True,
    fmt='d',
    cmap='YlGnBu'
)

plt.title(f'Improved PCA Confusion Matrix (Accuracy: {accuracy:.3f})')

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("pca_confusion_matrix.png")

plt.close()


# --------------------------------------------------
# Create ZIP Results
# --------------------------------------------------
print("Creating ZIP file...")

zip_folder = "PCA_Baseline_Results"

os.makedirs(zip_folder, exist_ok=True)

shutil.copy("traditional_metrics.json",
            os.path.join(zip_folder, "traditional_metrics.json"))

shutil.copy("pca_confusion_matrix.png",
            os.path.join(zip_folder, "pca_confusion_matrix.png"))

shutil.make_archive("PCA_Results", "zip", zip_folder)

print("ZIP file created")


print("Total execution time:", round(time.time() - start_time, 2), "seconds")