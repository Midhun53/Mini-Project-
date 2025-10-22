import pandas as pd
import numpy as np
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import pickle

fake = Faker()
data = []

for _ in range(10000):
    amount = round(random.uniform(5, 2000), 2)
    time_gap = random.randint(1, 72)
    country = random.choice(['US', 'IN', 'UK', 'CA', 'AU'])
    device = random.choice(['Mobile', 'Desktop', 'Tablet'])
    age = random.randint(18, 70)
    is_fraud = 1 if (amount > 1500 or (country != 'US' and amount > 1000) or time_gap < 2) else 0
    data.append([amount, time_gap, country, device, age, is_fraud])

df = pd.DataFrame(data, columns=['Amount', 'TimeGap', 'Country', 'Device', 'Age', 'Fraud'])
df = pd.get_dummies(df, columns=['Country', 'Device'], drop_first=True)

X = df.drop('Fraud', axis=1)
y = df['Fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred)*100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred)*100, 2), "%")
print("F1-Score:", round(f1_score(y_test, y_pred)*100, 2), "%")

with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

print("Model saved successfully as fraud_detection_model.pkl")
