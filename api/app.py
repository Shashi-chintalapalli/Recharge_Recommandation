from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and tools
model = load_model("/Users/shashichintalapalli/Desktop/Recharge_Recommandation/models/dl_vi_recharge_model.keras")
scaler = joblib.load("/Users/shashichintalapalli/Desktop/Recharge_Recommandation/models/scaler.pkl")
label_encoders = joblib.load("/Users/shashichintalapalli/Desktop/Recharge_Recommandation/models/label_encoders.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Collect form inputs
        user_input = {
            'data_usage_GB': float(request.form['data_usage_GB']),
            'call_minutes': int(request.form['call_minutes']),
            'sms_count': int(request.form['sms_count']),
            'avg_recharge': float(request.form['avg_recharge']),
            'recharge_time': request.form['recharge_time'],
            'city': request.form['city'],
            'user_type': request.form['user_type']
        }

        # Create DataFrame
        df = pd.DataFrame([user_input])

        # Encode categorical features
        for col in ['recharge_time', 'city', 'user_type']:
            df[col] = label_encoders[col].transform(df[col])

        # Feature Engineering
        df['data_per_recharge'] = df['data_usage_GB'] / df['avg_recharge']
        df['calls_per_recharge'] = df['call_minutes'] / df['avg_recharge']
        df['sms_per_recharge'] = df['sms_count'] / df['avg_recharge']

        # Scale
        X_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(X_scaled)
        predicted_class = int(np.argmax(prediction))
        predicted_label = label_encoders['recommended_plan'].inverse_transform([predicted_class])[0]
        return render_template("index.html", prediction=predicted_label)


    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
