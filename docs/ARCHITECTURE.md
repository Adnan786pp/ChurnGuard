# 🏗️ Architecture Design Patterns

## Component Breakdown Data Workflow

1. Raw Tabular Customer Signals Stream
2. build_features.py Transformer (Tenure Buckets, Service Counts)
3. train.py Optimization System (XGBoost Estimator Serialization)
4. models/churn_model_pipeline.joblib (Immutable Frozen Asset)
5. FastAPI REST Backend <--- (Basic Key Auth Verification) ---> Streamlit Portal
6. TreeSHAP Explainability Layer (Plain-English Risk Statements Output)

## Security Profiles & Operational Safeguards
- **Gateway Validation:** Request validation payload handled strictly via Pydantic model schemas.
- **Basic Authentication Framework:** Headers validated sequentially via protected X-API-KEY checks.
- **Logging Pipeline:** Structured log streams tracking inference payloads and runtime metadata.
