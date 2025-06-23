import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load
model = load_model("../models/dl_vi_recharge_model.h5")
scaler = joblib.load("../models/scaler.pkl")
encoders = joblib.load("../models/label_encoders.pkl")

# Input (example)
input_dict = {
    'data_usage_GB': 6,
    'call_minutes': 180,
    'sms_count': 15,
    'avg_recharge': 199,
    'recharge_time': 'morning',
    'city': 'Delhi',
    'user_type': 'prepaid'
}

# Preprocess
input_df = pd.DataFrame([input_dict])
for col in ['recharge_time', 'city', 'user_type']:
    input_df[col] = encoders[col].transform(input_df[col])

input_df['data_per_recharge'] = input_df['data_usage_GB'] / input_df['avg_recharge']
input_df['calls_per_recharge'] = input_df['call_minutes'] / input_df['avg_recharge']
input_df['sms_per_recharge'] = input_df['sms_count'] / input_df['avg_recharge']

# Scale
X = scaler.transform(input_df)

# Predict
pred = model.predict(X)
predicted_class = np.argmax(pred)
print("Predicted Recharge Plan:", predicted_class)
