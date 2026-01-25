## Machine Learning on Encrypted Data using Homomorphic Encryption (CKKS)
** Student: Sania Dinesh Joshi (40425922)
** Supervisor: Dr. Amir Sabbagh Molahosseini

## Project Summary
This project investigates privacy‑preserving machine learning using Fully Homomorphic Encryption (FHE).
The work is divided into two complementary parts:

Low‑level encrypted arithmetic using the Microsoft SEAL library (CKKS scheme)

High‑level encrypted machine learning inference using the Concrete‑ML framework

The goal is to evaluate the feasibility of performing ML inference on encrypted data, and to measure the trade‑offs between accuracy, runtime, and model complexity.

## Repository Structure
Folder	Description
cpp/myseal/	      C++ implementation using Microsoft SEAL (CKKS). Includes encrypted arithmetic and vector operations.
python/	          Early experiments using TenSEAL for encrypted inference.
encrypted_ml/	  Concrete‑ML experiments for encrypted logistic regression, decision trees, and XGBoost.
reports/	      Meeting notes, progress reports, and experimental results.
scripts/	      Helper scripts for parameter sweeps, benchmarking, and reproducibility.
notebooks/	      Jupyter notebooks, including the Concrete‑ML Credit Scoring baseline.

## Encrypted Machine Learning (Concrete‑ML)
This section evaluates encrypted inference using the Concrete‑ML framework.
Experiments include:

Logistic Regression (sklearn vs Concrete‑ML)

Decision Tree Classifier (sklearn vs Concrete‑ML)

XGBoost (sklearn vs Concrete‑ML)

Quantization effects (n_bits)

FHE vs simulated vs plaintext inference

## Key Findings
Logistic Regression under FHE achieves identical accuracy to plaintext (≈98%).

Encrypted inference introduces a 1000×–2000× slowdown, depending on quantization.

Decision Trees show poor scalability under FHE:

inference time grows exponentially with tree depth

accuracy does not necessarily improve with deeper trees

Quantization (n_bits) is the dominant factor controlling runtime and model feasibility.

These results highlight the fundamental trade‑off between privacy and computational cost in homomorphic encryption.

## How to Run the Concrete‑ML Experiments
1. Install dependencies
bash
pip install -r requirements.txt
2. Launch Jupyter
bash
jupyter notebook
3. Open the notebook
Navigate to:

Code
notebooks/CreditScoring.ipynb
4. Run the baseline
## The notebook evaluates:

sklearn models

Concrete‑ML quantized models

FHE inference (with configurable sample size for speed)

Results are stored in a pandas DataFrame for comparison.


## How to Build and Run (C++)
```bash
cd cpp/myseal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
build/Release/myseal.exe 2.5 -1.7 -2 0 1.5 3.2


