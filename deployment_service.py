import os
import sys
import time
from io import StringIO
import uvicorn
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import Counter, Summary, generate_latest, CollectorRegistry

# --- Configuration ---
# Kubernetes sets this environment variable via the Helm Chart
MODEL_PATH = os.environ.get("MODEL_PATH") 

# --- Prometheus Metrics Setup ---
# Use a new registry for clean metrics reporting in FastAPI
REGISTRY = CollectorRegistry()

# 1. Counter: Tracks the total number of prediction requests received
PREDICTION_REQUESTS = Counter(
    'prediction_requests_total', 
    'Total number of prediction requests.', 
    ['model_name', 'status'], 
    registry=REGISTRY
)

# 2. Summary: Tracks the request latency in seconds
PREDICTION_LATENCY = Summary(
    'prediction_latency_seconds', 
    'Time spent processing prediction requests.', 
    ['model_name'], 
    registry=REGISTRY
)

# 3. Placeholder for Inference Data (Optional, but good practice)
# You might use a specific custom metric (Gauge) to track data drift over time,
# but for basic operational monitoring, Counter and Summary are sufficient.

# --- FastAPI Setup ---
app = FastAPI(
    title="MLflow Model Serving API",
    description="Serves the dynamically loaded MLflow model version."
)

# Global variable to hold the loaded model
model = None
MODEL_NAME = os.environ.get("MODEL_NAME", "unknown-model") # Use environment variable or default

class PredictionInput(BaseModel):
    features: list[float]

# --- Startup and Health Checks ---

@app.on_event("startup")
async def load_model_on_startup():
    """Loads the model from the shared volume on startup."""
    global model
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model path not found: {MODEL_PATH}")
        raise RuntimeError(f"Required model artifact not found at {MODEL_PATH}")
    
    try:
        print(f"Attempting to load model from: {MODEL_PATH}")
        model = mlflow.pyfunc.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model from disk: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

@app.get("/health")
def health_check():
    """Kubernetes Liveness and Readiness Probe endpoint."""
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    else:
        raise HTTPException(status_code=503, detail="Model is not yet loaded or failed to load")

# --- Prediction Endpoint ---

@app.post("/predict")
@PREDICTION_LATENCY.labels(model_name=MODEL_NAME).time() # Measure latency for this call
def predict(data: PredictionInput):
    """Main endpoint for receiving input features and returning predictions."""
    if model is None:
        PREDICTION_REQUESTS.labels(model_name=MODEL_NAME, status='error').inc()
        raise HTTPException(status_code=503, detail="Model not initialized.")
    
    try:
        input_data = [data.features] 
        
        # Get the prediction
        prediction = model.predict(input_data)
        
        PREDICTION_REQUESTS.labels(model_name=MODEL_NAME, status='success').inc()
        
        # In a production system, you would log input data and prediction 
        # for ground truth joining (monitoring model drift)
        
        return {"prediction": prediction.tolist()}
        
    except Exception as e:
        print(f"Prediction error: {e}")
        PREDICTION_REQUESTS.labels(model_name=MODEL_NAME, status='error').inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")

# --- Metrics Endpoint for prometheus to scrape ---

@app.get("/metrics")
def metrics():
    """
    Prometheus scrape endpoint. Returns the latest metrics data.
    """
    # Generate the metrics data in Prometheus format
    return Response(content=generate_latest(REGISTRY), media_type="text/plain; version=0.0.4; charset=utf-8")

# --- Run the App ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)