import time
import json
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as PlainTree

from concrete.ml.sklearn import DecisionTreeClassifier as EncryptedTree


def main():
    print("=" * 60)
    print("ENCRYPTED DECISION TREE - CONCRETE-ML")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # keep small test set for speed
    X_test = X_test[:50]
    y_test = y_test[:50]

    # scaling helps HE (keeps values in a nicer range)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for max_depth in [2, 3, 5]:
        print(f"\n--- max_depth={max_depth} ---")

        # plaintext baseline
        plain = PlainTree(max_depth=max_depth, random_state=42)
        plain.fit(X_train, y_train)

        t0 = time.time()
        pred_plain = plain.predict(X_test)
        t_plain = time.time() - t0
        acc_plain = accuracy_score(y_test, pred_plain)

        # encrypted model
        enc = EncryptedTree(max_depth=max_depth, n_bits=8)
        enc.fit(X_train, y_train)

        t0 = time.time()
        enc.compile(X_train)
        t_compile = time.time() - t0

        t0 = time.time()
        pred_enc = enc.predict(X_test, fhe="execute")
        t_enc = time.time() - t0
        acc_enc = accuracy_score(y_test, pred_enc)

        print(f"Plain acc={acc_plain:.3f} | Enc acc={acc_enc:.3f}")
        print(f"Compile={t_compile:.3f}s | Encrypted infer total={t_enc:.3f}s | Slowdown={t_enc/max(t_plain,1e-9):.1f}x")

        results.append({
            "model": "DecisionTreeClassifier",
            "max_depth": max_depth,
            "n_bits": 8,
            "n_test": int(len(X_test)),
            "acc_plain": float(acc_plain),
            "acc_enc": float(acc_enc),
            "compile_s": float(t_compile),
            "plain_infer_total_s": float(t_plain),
            "enc_infer_total_s": float(t_enc),
            "slowdown": float(t_enc / max(t_plain, 1e-9)),
            "enc_infer_per_sample_ms": float(t_enc / len(X_test) * 1000),
        })

    out_file = out_dir / "tree_experiments.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved: {out_file}")


if __name__ == "__main__":
    main()
