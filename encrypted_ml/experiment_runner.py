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


def run_experiment(n_bits, n_test=50):
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

    # Plain model
    plain = PlainLR(max_iter=1000)
    plain.fit(X_train, y_train)

    t0 = time.time()
    y_plain = plain.predict(X_test)
    plain_time = time.time() - t0
    acc_plain = accuracy_score(y_test, y_plain)

    # Encrypted model
    enc = EncryptedLR(n_bits=n_bits)
    enc.fit(X_train, y_train)
    enc.compile(X_train)

    t0 = time.time()
    y_enc = enc.predict(X_test, fhe="execute")
    enc_time = time.time() - t0
    acc_enc = accuracy_score(y_test, y_enc)

    return {
        "n_bits": n_bits,
        "acc_plain": acc_plain,
        "acc_encrypted": acc_enc,
        "accuracy_drop": acc_plain - acc_enc,
        "plain_time_ms": plain_time / n_test * 1000,
        "encrypted_time_ms": enc_time / n_test * 1000,
        "slowdown": enc_time / max(plain_time, 1e-9)
    }


def main():
    results = []

    for n_bits in [6, 8, 10]:
        print(f"\nRunning experiment with n_bits={n_bits}")
        res = run_experiment(n_bits)
        results.append(res)

        print(
            f"Acc plain={res['acc_plain']:.3f}, "
            f"Acc enc={res['acc_encrypted']:.3f}, "
            f"Slowdown={res['slowdown']:.1f}x"
        )

    with open(RESULTS_DIR / "precision_tradeoff.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved results to precision_tradeoff.json")


if __name__ == "__main__":
    main()
