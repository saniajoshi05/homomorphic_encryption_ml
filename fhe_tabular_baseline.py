import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from concrete.ml.sklearn import XGBClassifier


def run_experiment(n_bits):
    # Load data
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model (tree-based, NOT linear)
    model = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        n_bits=n_bits,
    )

    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Plain inference
    t0 = time.time()
    y_plain = model.predict(X_test)
    plain_time = time.time() - t0
    acc_plain = accuracy_score(y_test, y_plain)

    # Compile to FHE
    t0 = time.time()
    model.compile(X_train)
    compile_time = time.time() - t0

    # FHE inference (small subset)
    X_fhe = X_test[:20]
    y_fhe_true = y_test[:20]

    t0 = time.time()
    y_fhe = model.predict(X_fhe, fhe="execute")
    fhe_time = time.time() - t0
    acc_fhe = accuracy_score(y_fhe_true, y_fhe)

    return {
        "n_bits": n_bits,
        "acc_plain": acc_plain,
        "acc_fhe": acc_fhe,
        "train_s": train_time,
        "compile_s": compile_time,
        "fhe_s": fhe_time,
    }


if __name__ == "__main__":
    results = []
    for n_bits in [6, 8, 10]:
        print(f"Running n_bits={n_bits}")
        results.append(run_experiment(n_bits))

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("baseline_tabular_results.csv", index=False)
