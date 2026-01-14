import time
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression as PlainLR
from concrete.ml.sklearn import LogisticRegression as EncryptedLR

print("=" * 60)
print("ENCRYPTED LOGISTIC REGRESSION - PROOF OF CONCEPT")
print("=" * 60)

# 1) Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Keep test small for first run (faster)
X_test = X_test[:50]
y_test = y_test[:50]

# Scale features (important for HE)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
print(f"Features: {X_train.shape[1]}")

# 2) Plain baseline
print("\n--- PLAINTEXT MODEL ---")
plain_model = PlainLR(max_iter=1000, random_state=42)

t0 = time.time()
plain_model.fit(X_train, y_train)
train_time_plain = time.time() - t0

t0 = time.time()
y_pred_plain = plain_model.predict(X_test)
infer_time_plain = time.time() - t0

acc_plain = accuracy_score(y_test, y_pred_plain)
print(f"Train time:   {train_time_plain:.4f}s")
print(f"Infer time:   {infer_time_plain:.4f}s")
print(f"Accuracy:     {acc_plain:.4f}")

# 3) Encrypted model
print("\n--- ENCRYPTED MODEL (Concrete-ML) ---")
n_bits = 8  # start small for speed

enc_model = EncryptedLR(n_bits=n_bits, random_state=42)

print("Training (plaintext training, FHE-ready model)...")
t0 = time.time()
enc_model.fit(X_train, y_train)
train_time_enc = time.time() - t0
print(f"Train time:   {train_time_enc:.4f}s")

print("Compiling for FHE (builds the encrypted circuit)...")
t0 = time.time()
enc_model.compile(X_train)
compile_time = time.time() - t0
print(f"Compile time: {compile_time:.4f}s")

print("Running encrypted inference (FHE execute)...")
t0 = time.time()
y_pred_enc = enc_model.predict(X_test, fhe="execute")
infer_time_enc = time.time() - t0

acc_enc = accuracy_score(y_test, y_pred_enc)
print(f"Infer time:   {infer_time_enc:.4f}s")
print(f"Accuracy:     {acc_enc:.4f}")

print("\n--- SUMMARY ---")
print(f"Plain acc:    {acc_plain:.4f}")
print(f"Enc acc:      {acc_enc:.4f}")
print(f"Acc drop:     {acc_plain - acc_enc:.4f}")
print(f"Slowdown:     {infer_time_enc / max(infer_time_plain, 1e-9):.1f}x")

results = {
    "dataset": "breast_cancer",
    "model": "LogisticRegression",
    "n_bits": n_bits,
    "n_train": int(X_train.shape[0]),
    "n_test": int(len(X_test)),
    "n_features": int(X_train.shape[1]),
    "acc_plain": float(acc_plain),
    "acc_enc": float(acc_enc),
    "train_time_plain_s": float(train_time_plain),
    "train_time_enc_s": float(train_time_enc),
    "compile_time_s": float(compile_time),
    "infer_time_plain_total_s": float(infer_time_plain),
    "infer_time_enc_total_s": float(infer_time_enc),
    "infer_time_plain_per_sample_ms": float(infer_time_plain / len(X_test) * 1000),
    "infer_time_enc_per_sample_ms": float(infer_time_enc / len(X_test) * 1000),
}

from pathlib import Path

out_dir = Path(__file__).resolve().parent / "results"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "day1_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved: {out_dir / 'day1_results.json'}")
