"""
Systematic experiments for encrypted machine learning using Concrete-ML.
Evaluates accuracy, performance, and precision trade-offs.
"""

import time
import json
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression as PlainLR
from concrete.ml.sklearn import LogisticRegression as EncryptedLR

RESULTS_DIR = Path("encrypted_ml/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(n_bits, n_test=50, noise_std=0.15):
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_test = X_test[:n_test]
    y_test = y_test[:n_test]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Make task harder so n_bits effect is visible (controlled noise)
    rng = np.random.default_rng(42)
    X_train = X_train + rng.normal(0, noise_std, X_train.shape)
    X_test = X_test + rng.normal(0, noise_std, X_test.shape)

    # Plain model (baseline)
    plain = PlainLR(max_iter=5000, C=2.0, solver="lbfgs")
    plain.fit(X_train, y_train)

    t0 = time.time()
    y_plain = plain.predict(X_test)
    plain_time = time.time() - t0
    acc_plain = accuracy_score(y_test, y_plain)

    # Encrypted model
    enc = EncryptedLR(n_bits=n_bits, random_state=42)
    enc.fit(X_train, y_train)
    enc.compile(X_train)

    t0 = time.time()
    y_enc = enc.predict(X_test, fhe="execute")
    enc_time = time.time() - t0
    acc_enc = accuracy_score(y_test, y_enc)

    return {
        "dataset": "breast_cancer",
        "n_test": n_test,
        "noise_std": noise_std,
        "n_bits": n_bits,
        "acc_plain": float(acc_plain),
        "acc_encrypted": float(acc_enc),
        "accuracy_drop": float(acc_plain - acc_enc),
        "plain_time_ms": float(plain_time / n_test * 1000),
        "encrypted_time_ms": float(enc_time / n_test * 1000),
        "slowdown": float(enc_time / max(plain_time, 1e-9)),
    }


def main():
    results = []

    for n_bits in [4, 6, 8, 10, 12]:
        print(f"\nRunning experiment with n_bits={n_bits}")
        res = run_experiment(n_bits)
        results.append(res)

        print(
            f"Acc plain={res['acc_plain']:.3f}, "
            f"Acc enc={res['acc_encrypted']:.3f}, "
            f"Drop={res['accuracy_drop']:.3f}, "
            f"Slowdown={res['slowdown']:.1f}x"
        )

    with open(RESULTS_DIR / "precision_tradeoff.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved results to precision_tradeoff.json")


if __name__ == "__main__":
    main()
