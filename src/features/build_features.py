import pandas as pd
import numpy as np

def generate_mock_telco_data(num_rows=1000, random_state=42):
    np.random.seed(random_state)
    data = {
        'customerID': [f'CUST-{i:05d}' for i in range(num_rows)],
        'SeniorCitizen': np.random.choice([0, 1], size=num_rows, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], size=num_rows),
        'Dependents': np.random.choice(['Yes', 'No'], size=num_rows, p=[0.7, 0.3]),
        'tenure': np.random.randint(1, 73, size=num_rows),
        'PhoneService': np.random.choice(['Yes', 'No'], size=num_rows, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], size=num_rows),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_rows),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], size=num_rows),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], size=num_rows),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], size=num_rows),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], size=num_rows),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], size=num_rows),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], size=num_rows),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_rows, p=[0.5, 0.25, 0.25]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], size=num_rows),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=num_rows),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, size=num_rows)
    }
    df = pd.DataFrame(data)
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges'] + np.random.normal(0, 10, num_rows)
    df['TotalCharges'] = df['TotalCharges'].clip(lower=18.25)
    
    churn_prob = 0.15 + (df['Contract'] == 'Month-to-month') * 0.3 + (df['MonthlyCharges'] > 70) * 0.2 - (df['tenure'] > 36) * 0.25
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    df['Churn'] = np.random.binomial(1, churn_prob)
    df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No'})
    return df

def engineer_features(df):
    df = df.copy()
    df['tenure_buckets'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 72], labels=['0-1 Year', '1-2 Years', '2-4 Years', '4-6 Years', '6+ Years'])
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services_subscribed'] = df[services].apply(lambda row: sum(row == 'Yes'), axis=1)
    df['charges_per_tenure_ratio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['contract_risk_flag'] = df['Contract'].apply(lambda x: 1 if x == 'Month-to-month' else 0)
    df['high_charges_low_tenure'] = ((df['MonthlyCharges'] > 70) & (df['tenure'] <= 12)).astype(int)
    return df

if __name__ == "__main__":
    df = generate_mock_telco_data()
    df_feat = engineer_features(df)
    os.makedirs("src/data", exist_ok=True)
    df_feat.to_csv("src/data/processed_churn.csv", index=False)
    print("✅ Mock dataset generated successfully!")
