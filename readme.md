# MLOps Retrain & Rollback

This repository serves as a comprehensive technical guide and implementation framework for MLOps (Machine Learning Operations) - Retrain and Rollback. <br>

This implementation requires orchestrating two distinct lifecycles: 
1. Model Lifecycle (managed by MLflow) and
2. Application Lifecycle (managed by Helm/Kubernetes), tied together by GitHub Actions.



---

## 💻 The Retrain Strategy

The core idea is to establish an automated feedback loop where your monitoring system detects the performance drop and triggers your existing GitHub Retraining Action via a webhook. <br>

### I.	Monitoring
#### •	Metric Collection: 
Expose a dedicated /metrics endpoint (using a Python library like prometheus_client: CollectorRegistry, generate_latest) in your model serving application (see file MLOPS\Retrain&Rollback\deployment_service.py). The service should calculate and expose key performance metrics (like accuracy, F1-score, or RMSE) over a rolling time window (e.g., the last 24 hours of predictions).

#### •	Prometheus: 
Deploy Prometheus to your Kubernetes cluster to scrape the /metrics endpoint of your model service periodically. Prometheus collects this time-series data.

#### •	Grafana: 
Use Grafana to visualize the data from Prometheus.

### II.	Define the Retraining Alert
Either you use Grafana or Prometheus Alertmanager to define the specific condition that must be met to trigger retraining. 

### III.	The Webhook Trigger (The Bridge)
This is the critical step that connects the monitoring system to your CI/CD pipeline. <br>

#### Step A: Configure GitHub Actions for Webhook Trigger
You need to have a retrain.yml which will accept an external trigger using the repository_dispatch event. <br>
Save this file as .github/workflows/retrain.yaml in your repository. <br>

Please see retrain.yml for more details.<br>

```
on:
  #... existing triggers (schedule, workflow_dispatch) ...
  repository_dispatch:
  types: [model-performance-drop] # This is the unique event name
```

#### Step B: Configure the Alert to use the Webhook
```
• Target URL: https://api.github.com/repos/<owner>/<repo>/dispatches
• Method: POST
• Headers:
•	Authorization: token YOUR_GITHUB_PAT (Use a Personal Access Token with repo scope).
•	Content-Type: application/json
• Payload (Body): The alert needs to send the specific event name.
{
  "event_type": "model-performance-drop",
  "client_payload": {
    "model_name": "fraud-detection-model",
    "trigger_reason": "Accuracy dropped below benchmark"
  }
}
```

### IV.	The "train" Script (src/train.py)
See file train.py

This script encapsulates the entire training process: data loading, model training, metric calculation, and logging everything to MLflow. Crucially, it prints the **MLflow Run ID** in a format that **retrain.yml** workflow can easily capture and pass to the **gatekeeper.py** script.
**Integration Notes:**
In your train.py you have below line <br>
```
# --- CRITICAL OUTPUT FOR CI/CD --- # This print statement is parsed by the GitHub Actions workflow (retrain.yml)
# to get the Run ID and pass it to the gatekeeper script.
print(f"RUN_ID: {run_id}")
```
**retrain.yml** uses a specific command to capture this output and store it as an environment variable **($GITHUB_ENV):**

```
# Snippet from retrain.yml's 'Run Training' step
      - name: 🚀 Run Training and Log to MLflow
        id: training
        # ... setup ...
        run: |
          # 1. Execute the script and capture all output
          python src/train.py > run_output.txt
          
          # 2. Extract the RUN_ID by grepping the specific string and parsing it
          RUN_ID=$(grep 'RUN_ID' run_output.txt | cut -d ':' -f 2 | tr -d '[:space:]')
          
          # 3. Store the RUN_ID in GitHub Actions environment
          echo "RUN_ID=$RUN_ID" >> $GITHUB_ENV
```


### V.	The "Gatekeeper" Script (src/gatekeeper.py)
The gatekeeper.py script then uses this captured RUN_ID mentioned above to find the logged metrics and artifacts in MLflow for comparison. <br>

Please see the file gatekeeper.py


## 🚀 The Deployment Strategy (The Bridge)
Here is the implementation of the **Helm Chart template** designed for dynamic model loading.
This approach uses the Kubernetes **InitContainer** Pattern. <br>
This is the industry-standard way to separate your Application Code (Docker Image) from your Model Artifacts (Binary files).<br>

Please see the file deployment.yml <br>

The Architecture: **"Init-and-Mount"** <br>
Instead of baking the huge model file into your Docker image (which makes the image heavy and slow to build), we download the model at runtime. <br>

**Init Container:** Starts first. Connects to MLflow, downloads the specific model version (passed by GitHub Actions) to a shared volume, and then terminates. 

```
      echo "Downloading model version {{ .Values.model.version }}..."
      # MLflow CLI command to download artifacts to the shared volume
      mlflow artifacts download \
      --artifact-uri "models:/{{ .Values.model.name }}/{{ .Values.model.version }}" \
                --dst-path "/shared-data/model"


```

**Shared Volume:** A temporary storage space inside the Pod shared between containers. <br>

**Main Container:** Starts only after the Init Container finishes. It loads the model from the Shared Volume and serves the API.

1)	The Deployment Template (templates/deployment.yaml)
   This file defines the logic. Note the use of ```{{ .Values.model.version }}``` which allows GitHub Actions to inject the version number dynamically.
2)	The Values File (values.yaml)
   This acts as the default configuration. Your GitHub Action will override specific lines here.
3)	Tying it back to GitHub Actions
Now, look at how the deploy step in your GitHub Action actually injects the data into the template above. <br>
In your .github/workflows/retrain.yml:

```
     - name: ⬆️ Helm Upgrade and Deployment
        run: |
          # The core deployment command that injects the new model version
          helm upgrade --install ${{ env.HELM_RELEASE_NAME }} ./helm-chart \
            --namespace ${{ env.K8S_NAMESPACE }} \
            --set model.version=${{ needs.train-and-validate.outputs.model_version }} \
            --set model.mlflowUri=${{ secrets.MLFLOW_TRACKING_URI }} \
            --atomic \ # CRITICAL: Automatically rolls back if K8s health checks fail
            --timeout 5m \
            --wait # Wait for the deployment to complete and pass health checks

```

##  📈 The Rollback Strategy

**Scenario A: Deployment Failure (CrashLoopBackOff)** <br>
Detection: Kubernetes Liveness probes fail.
* Strategy: Automated via Helm.
* Implementation: In your helm upgrade command (above), the --atomic flag ensures that if the deployment doesn't reach a "Ready" state within the timeout, Helm essentially runs helm rollback automatically.

**Scenario B: Model Performance degradation (Logic Failure)** <br>
The app runs, but predictions are wrong (e.g., accuracy drops).
* Detection: Monitoring system (Prometheus/Grafana) detects drift or high error rates.
* Strategy: Automated GitHub Action Trigger.

1. The Rollback Workflow Create .github/workflows/rollback.yaml. <br>

See the file rollback.yml <br>
```
on:
  repository_dispatch:
    types: [performance-alert] # Triggered by Prometheus/Grafana webhook


helm rollback my-model-service 0 --namespace mlops-prod

```

2. MLflow Demotion Script (src/demote_model.py) You must ensure the registry reflects reality. <br>

See the file demote_model.py


## 📌 The Model Serving Application
This application code assumes you used mlflow.pyfunc.log_model during your training phase, making loading straightforward and robust. <br>

Please see deployment_service.py

## Key Integration Points
```
1.	MODEL_PATH: The line MODEL_PATH = os.environ.get("MODEL_PATH") directly reads the environment variable that was set in your Helm Deployment (deployment.yaml). This variable points to the shared volume path (/shared-data/model).
2.	load_model_on_startup: This function attempts to load the model. If the model file is corrupt (or the path is wrong), the function raises a RuntimeError. This is crucial because Kubernetes will then restart the container. If the container repeatedly fails to start (CrashLoopBackOff), the Helm --atomic deployment will automatically trigger a rollback to the last stable release.
3.	/health Endpoint: The health_check function provides a reliable status for the Liveness Probe. It confirms that the Python application is running AND that the model has successfully been loaded into memory (model is not None).
This script completes the circuit: the model version is specified by GitHub Actions, downloaded by the InitContainer, and consumed by this Main Application Script.
```

