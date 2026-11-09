# 📉 Monitoring & Maintenance Framework

## Data Drift Analysis Guidelines
To maintain production alignment, check the statistical distributions of features monthly:
- **Numerical Trackers:** Run a Kolmogorov-Smirnov test on MonthlyCharges and charges_per_tenure_ratio.
- **Categorical Trackers:** Run a Population Stability Index (PSI) analysis on Contract and InternetService vectors. Target critical thresholds: PSI > 0.2 indicates severe feature shift.

## Retraining Guidelines Conceptual Architecture
- **Retraining Trigger Criteria:** A consistent drop in production Precision/Recall benchmarks lasting more than 14 business cycles or a structural drift threshold spike inside categorical columns.
- **Pipeline Re-execution Process:** Feed newly captured logs data directly through build_features.py into train.py using automated orchestration layers.
