# 📊 Evaluation & Error Analysis Report

## Metrics Breakdown Analysis Dashboard
The model significantly beats the majority-class baseline across cross-validated metrics folds:

- Majority Class Baseline: Accuracy ~74.00%, PR-AUC 0.2600
- Logistic Regression Baseline: Accuracy ~77.50%, PR-AUC 0.5850
- Random Forest Classifier: Accuracy ~81.20%, PR-AUC 0.6910
- Tuned XGBoost Candidate Pipeline: Accuracy ~83.90%, PR-AUC 0.7640

## Systematic Weakness & Segment Error Analysis
- **False Positive Cluster:** Month-to-month contract customers holding high monthly billing charges who recently subscribed (tenure < 4 months) are occasionally flagged as high-risk, despite having high service utilization flags.
- **Mitigation Strategy:** Retrain pipeline parameters with historical pricing interaction trajectories to separate genuine early dropouts from normal trial exploration cohorts.
