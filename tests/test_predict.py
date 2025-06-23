# tests/test_predict.py

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load components from models/
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
model = load_model("models/dl_vi_recharge_model.h5")

def test_prediction():
    # Sample input user data
    sample_input = {
        'data_usage_GB': 6,
        'call_minutes': 200,
        'sms_count': 15,
        'avg_recharge': 199,
        'recharge_time': 'morning',
        'city': 'Delhi',
        'user_type': 'prepaid'
    }

    df = pd.DataFrame([sample_input])

    # Encode categorical variables
    for col in ['recharge_time', 'city', 'user_type']:
        df[col] = encoders[col].transform(df[col])

    # Feature engineering
    df['data_per_recharge'] = df['data_usage_GB'] / df['avg_recharge']
    df['calls_per_recharge'] = df['call_minutes'] / df['avg_recharge']
    df['sms_per_recharge'] = df['sms_count'] / df['avg_recharge']

    # Scale features
    X_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(X_scaled)
    predicted_plan = int(np.argmax(prediction))

    # Assert result is within expected class range
    assert predicted_plan in [0, 1, 2], f"Invalid prediction: {predicted_plan}"

    print("✅ Test passed. Predicted recharge plan:", predicted_plan)
