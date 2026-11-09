# 🛡️ ChurnGuard: Customer Churn Prediction & Retention Support System

ChurnGuard is a deployed, production-grade Machine Learning solution designed for subscription-based businesses (e.g., Telecom) to proactively identify at-risk customers, understand local risk drivers using SHAP explainability mechanics, and triage customer retention workflows effectively.

## 🚀 Quick Start & Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (Optional)

### Installation
1. Clone the repository and navigate to the root directory:
```bash
git clone https://github.com
cd ChurnGuard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
1. Run Feature Engineering & Mock Data Layer Builder:
```bash
python src/features/build_features.py
```

2. Train and Serialize the XGBoost Model Pipeline:
```bash
python src/models/train.py
```

3. Run Production API Testing Suite:
```bash
python -m pytest tests/test_api.py -v
```

4. Launch the Production FastAPI Application Server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

5. Launch the Streamlit Portal:
```bash
streamlit run src/frontend/app.py
```
