import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Get base path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
data_path = os.path.join(BASE_DIR, '../Data/vi_recharge1.csv')
df = pd.read_csv(data_path)

# Feature Engineering
df['data_per_recharge'] = df['data_usage_GB'] / df['avg_recharge']
df['calls_per_recharge'] = df['call_minutes'] / df['avg_recharge']
df['sms_per_recharge'] = df['sms_count'] / df['avg_recharge']

# Label encoding
label_cols = ['last_recharge', 'recharge_time', 'city', 'user_type', 'recommended_plan']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop columns not needed
df = df.drop(['last_recharge', 'user_id'], axis=1)

# Split features and target
X = df.drop("recommended_plan", axis=1)
y = df["recommended_plan"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------- Random Forest Model -------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

print("Random Forest Classification Report:\n", classification_report(y_test, rf_preds))

# ----------------- Deep Learning Model -------------------
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

dl_model.fit(X_train_scaled, y_train_cat, epochs=2, batch_size=32, validation_split=0.2)

loss, accuracy = dl_model.evaluate(X_test_scaled, y_test_cat)
print(f"Deep Learning Accuracy: {accuracy:.2f}")

# ----------------- Save Models & Encoders -------------------
model_dir = os.path.join(BASE_DIR, '../models')
os.makedirs(model_dir, exist_ok=True)

dl_model.save(os.path.join(model_dir, 'dl_vi_recharge_model.keras'))
joblib.dump(rf_model, os.path.join(model_dir, 'rf_vi_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))

print("✅ Models and encoders saved successfully.")
