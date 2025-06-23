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

# Load CSV file
df = pd.read_csv('/Users/shashichintalapalli/Desktop/Recharge_Recommandation/data/vi_recharge1.csv')


# Feature Engineering
df['data_per_recharge'] = df['data_usage_GB'] / df['avg_recharge']
df['calls_per_recharge'] = df['call_minutes'] / df['avg_recharge']
df['sms_per_recharge'] = df['sms_count'] / df['avg_recharge']

# Label encoding
label_col = ['last_recharge','recharge_time','city','user_type','recommended_plan']
label_encoders = {}
for col in label_col:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop unnecessary columns
df = df.drop(['last_recharge', 'user_id'], axis=1)

# Split into features and target
X = df.drop("recommended_plan", axis=1)
y = df["recommended_plan"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForest model
rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
print("RandomForest Report:\n", classification_report(y_test, rf_preds))

# Deep Learning model
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

dl_model = Sequential()
dl_model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
dl_model.add(Dropout(0.2))
dl_model.add(Dense(64, activation='relu'))
dl_model.add(Dropout(0.2))
dl_model.add(Dense(3, activation='softmax'))

dl_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
dl_model.fit(X_train_scaled, y_train_cat, epochs=2, batch_size=32, validation_split=0.2)
loss, acc = dl_model.evaluate(X_test_scaled, y_test_cat)
print(f"Deep Learning Accuracy: {acc:.2f}")

# Save everything
dl_model.save("models/dl_vi_recharge_model.keras", save_format="keras")
joblib.dump(rf_model, "models/rf_vi_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")