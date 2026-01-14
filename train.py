import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# --- Configuration ---
# Set the MLflow experiment name
MLFLOW_EXPERIMENT_NAME = "FraudDetectionTraining"
MODEL_ARTIFACT_PATH = "model"

def load_data():
    """Simulates loading and preprocessing data."""
    print("Loading and preparing data...")
    try:
        # In a real scenario, this would load data from a database or S3 bucket
        # Placeholder: Generate dummy data for illustration
        X = pd.DataFrame(np.random.rand(500, 10), columns=[f'f{i}' for i in range(10)])
        y = pd.Series(np.random.randint(0, 2, 500))
        
        # Simulate data drift/preprocessing if necessary
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and calculates key metrics."""
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }
    
    print(f"Metrics: {metrics}")
    return metrics

def train_model():
    """Main function to train, log, and output the MLflow run details."""
    
    # MLflow automatically reads the MLFLOW_TRACKING_URI environment variable
    # set by the GitHub Action
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    X_train, X_test, y_train, y_test = load_data()

    # --- Start MLflow Run ---
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run started with ID: {run_id}")

        # Define model parameters (can be read from config or command line)
        n_estimators = 100
        max_depth = 5
        
        # 1. Log Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # 2. Train Model
        print("Starting model training...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        print("Model training complete.")
        
        # 3. Evaluate and Log Metrics
        metrics = evaluate_model(model, X_test, y_test)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            
        # 4. Log Model Artifact
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path=MODEL_ARTIFACT_PATH,
            # This allows the model to be loaded by mlflow.pyfunc later
            registered_model_name=None # Registering happens in gatekeeper.py
        )
        print(f"Model artifacts logged to path: {MODEL_ARTIFACT_PATH}")
        
        # --- CRITICAL OUTPUT FOR CI/CD ---
        # This print statement is parsed by the GitHub Actions workflow (.yaml)
        # to get the Run ID and pass it to the gatekeeper script.
        print(f"RUN_ID: {run_id}") 
        
    return run_id

if __name__ == "__main__":
    train_model()