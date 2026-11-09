from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
import uvicorn
import logging
import joblib
from src.models.explainability import get_local_explanations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ChurnGuardAPI")

app = FastAPI(title="ChurnGuard Engine")

API_KEY = "churn_guard_secure_token_2026"
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key authentication credentials provided.")
    return api_key

class CustomerDataPayload(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_churn(payload: CustomerDataPayload):
    try:
        # Convert incoming payload dictionary structural entries to dataframe
        input_data = pd.DataFrame([payload.model_dump()])
        
        # Inject required dynamic feature engineering layer transformations inline to guarantee mapping matches
        input_data['tenure_buckets'] = pd.cut(input_data['tenure'], bins=[0, 12, 24, 48, 72, 100], labels=['0-1 Year', '1-2 Years', '2-4 Years', '4-6 Years', '6+ Years'])
        services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        input_data['total_services_subscribed'] = input_data[services].apply(lambda row: sum(row == 'Yes'), axis=1)
        input_data['charges_per_tenure_ratio'] = input_data['MonthlyCharges'] / (input_data['tenure'] + 1)
        input_data['contract_risk_flag'] = input_data['Contract'].apply(lambda x: 1 if x == 'Month-to-month' else 0)
        input_data['high_charges_low_tenure'] = ((input_data['MonthlyCharges'] > 70) & (input_data['tenure'] <= 12)).astype(int)
        
        insights = get_local_explanations(input_data)
        
        return {
            "success": True,
            "churn_probability": round(insights["churn_probability"], 4),
            "risk_status": insights["risk_level"],
            "actionable_insights": insights["top_drivers"]
        }
    except Exception as e:
        logger.error(f"Prediction Pipeline failure trace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal pipeline analytical execution exception: {str(e)}")
