## Machine Learning on Encrypted Data using Homomorphic Encryption (CKKS)
Student: Sania Dinesh Joshi (40425922)
Supervisor: Dr. Amir Sabbagh Molahosseini

## Project Overview
This project explores privacy‑preserving machine learning using Fully Homomorphic Encryption (FHE). The work is divided into two complementary components:

Low‑level encrypted computation using the Microsoft SEAL library (CKKS scheme), focusing on encrypted arithmetic and vector operations.

High‑level encrypted machine learning inference using the Concrete‑ML framework, evaluating practical ML models under FHE constraints.

The overarching goal is to assess the feasibility of performing machine‑learning inference directly on encrypted data and to quantify the trade‑offs between accuracy, runtime, and model complexity.

## Repository Structure
Folder	                 Description
cpp/myseal/	       C++ implementation using Microsoft SEAL (CKKS). Includes encrypted arithmetic, vector operations, and reproducible examples.
python/	Early      experiments using TenSEAL for encrypted inference.
encrypted_ml/	   Concrete‑ML experiments for encrypted Logistic Regression, Decision Trees, and XGBoost.
reports/	       Meeting notes, progress summaries, and experimental logs.
scripts/	       Helper scripts for benchmarking, parameter sweeps, and reproducibility.
notebooks/	       Jupyter notebooks, including the Concrete‑ML Credit Scoring baseline and extended experiments.

## Encrypted Machine Learning (Concrete‑ML)
This section evaluates encrypted inference using the Concrete‑ML framework.
Experiments include:

Logistic Regression (sklearn vs Concrete‑ML)

Decision Tree Classifier (sklearn vs Concrete‑ML)

XGBoost (sklearn vs Concrete‑ML)

Quantization effects (n_bits)

Comparison of plaintext, simulated FHE, and real FHE inference

These experiments measure how model structure and quantization influence encrypted inference performance.

## Key Findings

## Logistic Regression
Concrete‑ML’s quantized Logistic Regression achieves identical accuracy to plaintext (~98%).

FHE inference is 1000×–2000× slower, depending on quantization precision.

## Decision Trees
Decision Trees scale poorly under FHE:

Runtime increases sharply with tree depth.

Accuracy does not consistently improve with deeper trees.

Quantization (n_bits) is the dominant factor affecting both runtime and feasibility.

## General Insight
FHE enables privacy‑preserving inference, but introduces significant computational overhead.
The results highlight a core trade‑off: higher privacy → higher computational cost.

## Running the Concrete‑ML Experiments
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
The notebook evaluates:
sklearn models
Concrete‑ML quantized models
FHE inference (with configurable sample size for speed)

All results are collected into a pandas DataFrame for comparison, and plots are generated for runtime and accuracy analysis.

## Building and Running the C++ CKKS Examples
bash
cd cpp/myseal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
build/Release/myseal.exe 2.5 -1.7 -2 0 1.5 3.2
This runs encrypted arithmetic operations using the CKKS scheme implemented with Microsoft SEAL.