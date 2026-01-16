# Machine Learning on Encrypted Data using Homomorphic Encryption (CKKS)

**Student:** Sania Dinesh Joshi (40425922)  
**Supervisor:** Dr. Amir Sabbagh Molahosseini  

## Project Summary
This project explores privacy-preserving Machine Learning using the Microsoft SEAL library (CKKS scheme).  
The goal is to perform encrypted arithmetic and measure performance–accuracy trade-offs when training or evaluating models on encrypted data.

## Repository Structure
- `cpp/myseal/` → C++ implementation using Microsoft SEAL  
- `python/` → initial experiments with the TenSEAL Python library  
- `reports/` → meeting notes, progress reports, and results  
- `scripts/` → helper scripts for parameter testing and benchmarking

## Encrypted Machine Learning (Concrete-ML)

This folder contains experiments using Concrete-ML to evaluate
machine learning inference on encrypted data.

### Files
- `first_encrypted_model.py`  
  Proof-of-concept encrypted logistic regression using CKKS-style FHE.

- `results/`  
  Stores experimental outputs such as accuracy and timing results.

### Purpose
This work extends earlier Microsoft SEAL experiments by using
a higher-level encrypted ML framework to evaluate:
- accuracy impact of encryption
- performance overhead
- feasibility of encrypted ML inference in practice


## Encrypted ML Experiments (Concrete-ML)

- Logistic Regression under FHE shows identical accuracy to plaintext (≈98%)
- Encrypted inference incurs 1000×–2000× slowdown
- Decision Trees show severe scalability issues:
  - Encrypted inference time grows rapidly with tree depth
  - Accuracy does not necessarily improve with complexity

These results highlight the trade-off between privacy and computational cost in homomorphic encryption.


## How to Build and Run (C++)
```bash
cd cpp/myseal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
build/Release/myseal.exe 2.5 -1.7 -2 0 1.5 3.2


