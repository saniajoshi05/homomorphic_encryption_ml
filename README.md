# Machine Learning on Encrypted Data using Homomorphic Encryption (CKKS)

**Student:** Sania Dinesh Joshi (40425922)  
**Supervisor:** Dr. Amir Sabbagh Molahosseini  

## Project Overview

This project investigates privacy-preserving machine learning using Fully Homomorphic Encryption (FHE), focusing on both low-level encrypted computation and high-level encrypted model inference.
The work is divided into two complementary components:

### Low-Level Encrypted Computation (Microsoft SEAL – CKKS)

- Encrypted arithmetic and vector operations in C++
- Exploration of CKKS approximate arithmetic
- Performance analysis of encrypted numerical operations

### High-Level Encrypted Machine Learning (Concrete-ML)

- Encrypted inference for practical ML models
- Logistic Regression, Decision Trees, and XGBoost
- Evaluation of quantization and model complexity under FHE
- Comparison of plaintext, simulated FHE, and real FHE execution

### Project Goal

To assess the feasibility of running machine-learning inference directly on encrypted data and to quantify the trade-offs between:
- Accuracy  
- Runtime  
- Model complexity  
- Privacy  

## Repository Structure

| Folder              | Description                                                                              |
| `cpp/myseal/`       | C++ experiments using Microsoft SEAL (CKKS): encrypted arithmetic and vector operations  |
| `python/`           | Early experiments using TenSEAL                                                          |
| `encrypted_ml/`     | Concrete-ML experiments (Logistic Regression, Decision Trees, XGBoost)                   |
| `notebooks/`        | Jupyter notebooks including the Concrete-ML Credit Scoring baseline and extensions       |
| `scripts/`          | Helper scripts for benchmarking and parameter sweeps                                     |
| `reports/`          | Meeting notes, progress summaries, and experimental logs                                 |


## Encrypted Machine Learning (Concrete-ML)

This section evaluates encrypted inference using the Concrete-ML framework.

Experiments include:
- Logistic Regression (sklearn vs Concrete-ML)
- Decision Tree Classifier (sklearn vs Concrete-ML)
- XGBoost (sklearn vs Concrete-ML)
- Quantization effects (`n_bits`)
- Comparison of plaintext, simulated FHE, and real FHE inference

These experiments analyse how model structure and quantization precision affect encrypted inference performance.


## Key Findings

### Logistic Regression

- Concrete-ML achieves identical accuracy to plaintext (~98%).
- FHE inference is approximately 1000×–2000× slower, depending on quantization.

### Decision Trees and Tree-Based Models

- Runtime increases sharply with tree depth.
- Accuracy does not consistently improve with deeper trees.
- Quantization (`n_bits`) is the dominant factor affecting both feasibility and performance.

### General Insight

Fully Homomorphic Encryption enables privacy-preserving inference, but introduces significant computational overhead.
This highlights a fundamental trade-off:
Stronger privacy leads to higher computational cost.

## Running the Concrete-ML Experiments

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Launch Jupyter
jupyter notebook
3. Open the notebook
Navigate to:

notebooks/CreditScoring.ipynb
4. Run the baseline
The notebook evaluates:
sklearn models
Concrete-ML quantized models
FHE inference (with configurable sample size for faster experimentation)
Results are collected into pandas DataFrames and visualised with runtime and accuracy plots.

Building and Running the C++ CKKS Examples (Microsoft SEAL)
cd cpp/myseal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
build/Release/myseal.exe 2.5 -1.7 -2 0 1.5 3.2
This executes encrypted arithmetic operations using the CKKS scheme implemented with Microsoft SEAL.

