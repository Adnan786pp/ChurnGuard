import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)
API_KEY = "churn_guard_secure_token_2026"

@pytest.fixture
def valid_payload():
    return {
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5
    }

def test_health_endpoint():
    """Verify system operational performance check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_happy_path(valid_payload):
    """Verify endpoint processes valid structural payload securely."""
    headers = {"X-API-KEY": API_KEY}
    response = client.post("/predict", json=valid_payload, headers=headers)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert "churn_probability" in json_data
    assert "risk_status" in json_data
    assert len(json_data["actionable_insights"]) > 0

def test_predict_auth_failure(valid_payload):
    """Verify incorrect credentials block service access authorization."""
    headers = {"X-API-KEY": "wrong_token_value"}
    response = client.post("/predict", json=valid_payload, headers=headers)
    assert response.status_code == 403
    assert "Invalid API Key" in response.json()["detail"]

def test_predict_invalid_payload_types(valid_payload):
    """Verify engine constraints catch corrupted fields attributes validation types."""
    headers = {"X-API-KEY": API_KEY}
    corrupted_payload = valid_payload.copy()
    corrupted_payload["MonthlyCharges"] = "not-a-float-number"
    
    response = client.post("/predict", json=corrupted_payload, headers=headers)
    assert response.status_code == 422 # Unprocessable Entity Validation

