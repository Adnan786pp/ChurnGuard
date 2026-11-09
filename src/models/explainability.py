import shap
import joblib
import pandas as pd
import numpy as np

def get_local_explanations(customer_data, model_pipeline_path="models/churn_model_pipeline.joblib"):
    pipeline = joblib.load(model_pipeline_path)
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']
    
    X_transformed = preprocessor.transform(customer_data)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features = list(cat_encoder.get_feature_names_out())
    num_features = preprocessor.transformers_[0][2]
    all_features = num_features + cat_features
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    prob = pipeline.predict_proba(customer_data)[:, 1]
    customer_shap = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    
    top_indices = np.argsort(np.abs(customer_shap))[-3:][::-1]
    plain_english_drivers = []
    
    for idx in top_indices:
        feat_name = all_features[idx]
        shap_val = customer_shap[idx]
        direction = "increases risk" if shap_val > 0 else "decreases risk"
        clean_name = feat_name.replace("cat__", "").replace("num__", "").replace("_", " ")
        plain_english_drivers.append(f"• Configured '{clean_name}' value which {direction}.")
        
    return {
        "churn_probability": float(prob[0]),
        "risk_level": "High Risk" if prob[0] >= 0.4 else "Low/Managed Risk",
        "top_drivers": plain_english_drivers
    }
