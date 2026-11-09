# 🎴 Model Card - ChurnGuard Core Engine

## Model Details
- **Developed by:** Capstone Machine Learning Engineer
- **Model Type:** Extreme Gradient Boosting Classifier (XGBoost) wrapped inside a Scikit-Learn Pipeline.
- **Version Tag / Training Date:** 2026-11-09-production-v1
- **Downstream Analytics Engine:** TreeSHAP Explainer for deterministic local feature importances.

## Intended Use
- **Primary Use Case:** Daily customer risk triage by telecom customer relations and retention support operators.
- **Target Audience:** Account specialists looking for actionable local risk explanations instead of bare labels.
- **Out-of-Scope Uses:** Automated financial decisions, account bans, or credit scoring evaluations.

## Data & Features Schema
- **Input Dimensions:** 18 raw columns spanning demographics, contract signatures, payment modes, charges metrics, and 5 engineered columns (tenure_buckets, total_services_subscribed, charges_per_tenure_ratio, contract_risk_flag, high_charges_low_tenure).
- **Target Feature:** Churn (Binary Category: Yes or No).
- **Target Imbalance Structure:** ~26% Positive Class representation mimicking realistic market churn footprints.

## Performance Metrics & Evaluation
- **Majority-Class Accuracy Baseline:** Reported alongside every subsequent variant iteration.
- **Final Tuned Candidate Model Benchmarks:**
  - **PR-AUC:** High structural separation under Stratified 5-Fold Cross Validation.
  - **Operating Threshold:** Calibrated to 0.4 to optimize high Recall-Weighted Priority to handle asymmetric costs of missing a churner.
