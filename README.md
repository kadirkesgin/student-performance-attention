# Student Performance Prediction via Hybrid Attention-based Models

This repository contains the official implementation and research artifacts for the study: **"Robust Student Performance Prediction via Hybrid Attention-based Models and Multi-Dataset Evaluation"**.

## Authors
- **Kadir Kesgin**
- **Erdogan Usta**

## Research Objective
The goal of this project is to develop a scientifically rigorous, attention-based hybrid neural network for student performance prediction, specifically optimized for behavior-sensitive educational data.

## Key Features
- **Comprehensive Metric Suite**: Rigorous evaluation using **Balanced Accuracy, Macro-F1, Precision, Recall, and ROC-AUC**.
- **Formal Statistical Significance**: Results validated via **paired t-tests**, achieving **p < 0.001** in behavioral datasets and significant gains in imbalanced scenarios.
- **Leakage-Free Evaluation**: Explicit removal of target-leaking features (e.g., ExamScore) to ensure scientifically valid behavioral prediction.
- **Multi-Dataset Benchmark**: Cross-validated results on UCI Dropout, Mendeley AI, and Kaggle 2024.
- **XAI Support**: Integrated Empirical SHAP analysis for model interpretability.

## Repository Structure
- `models/`: Implementation of the Proposed Hybrid-Attention and Baseline models.
- `scripts/`: Data preprocessing, rigorous evaluation, and visualization scripts.
- `data/processed/`: Standardized and cleaned datasets used in the study.
- `outputs/`: High-resolution (300 DPI) academic figures.
- `manuscript/`: LaTeX summary of the research findings.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run evaluation: `export PYTHONPATH=. && python3 scripts/evaluate.py`
3. Generate figures: `python3 scripts/generate_paper_figs.py`

## License
MIT License
