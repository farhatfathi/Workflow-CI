# modelling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import mlflow
import mlflow.sklearn

# 1. Argument parsing dari MLproject
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

# 2. Load Preprocessed Dataset
link = 'https://raw.githubusercontent.com/farhatfathi/Eksperimen_SML_Muhammad-Fathi-Farhat/refs/heads/main/preprocessing/df_preprocessed.csv'
df = pd.read_csv(link)
df = df.drop('CustomerID', axis=1)

# 3. Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Logging & modelling langsung (tanpa start_run)
mlflow.set_experiment("churn-prediction")

model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Logging param, metric, dan model
mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")

print(f"Akurasi model: {acc}")
