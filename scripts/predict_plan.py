import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Define base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing tools
model_path = os.path.join(BASE_DIR, "../models/dl_vi_recharge_model.keras")
scaler_path = os.path.join(BASE_DIR, "../models/scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "../models/label_encoders.pkl")

model = load_model(model_path)
scaler = joblib.load(scaler_path)
encoders = joblib.load(encoder_path)

# Example input dictionary
input_dict = {
    'data_usage_GB': 6,
    'call_minutes': 180,
    'sms_count': 15,
    'avg_recharge': 199,
    'recharge_time': 'morning',
    'city': 'Delhi',
    'user_type': 'prepaid'
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])

# Encode categorical features
for col in ['recharge_time', 'city', 'user_type']:
    input_df[col] = encoders[col].transform(input_df[col])

# Feature engineering
input_df['data_per_recharge'] = input_df['data_usage_GB'] / input_df['avg_recharge']
input_df['calls_per_recharge'] = input_df['call_minutes'] / input_df['avg_recharge']
input_df['sms_per_recharge'] = input_df['sms_count'] / input_df['avg_recharge']

# Scale features
X = scaler.transform(input_df)

# Predict using the DL model
prediction = model.predict(X)
predicted_class = np.argmax(prediction)
predicted_label = encoders['recommended_plan'].inverse_transform([predicted_class])[0]

print("✅ Predicted Recharge Plan:", predicted_label)
