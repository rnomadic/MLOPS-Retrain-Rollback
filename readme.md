# MLOps Retrain & Rollback

This repository serves as a comprehensive technical guide and implementation framework for MLOps (Machine Learning Operations) - Retrain and Rollback. <br>

This implementation requires orchestrating two distinct lifecycles: 
1. Model Lifecycle (managed by MLflow) and
2. Application Lifecycle (managed by Helm/Kubernetes), tied together by GitHub Actions.



---

## 💻 The Retrain Strategy

The core idea is to establish an automated feedback loop where your monitoring system detects the performance drop and triggers your existing GitHub Retraining Action via a webhook. <br>

### I.	Monitoring
•	Metric Collection: Expose a dedicated /metrics endpoint (using a Python library like prometheus_client: CollectorRegistry, generate_latest) in your model serving application (see file MLOPS\Retrain&Rollback\deployment_service.py). The service should calculate and expose key performance metrics (like accuracy, F1-score, or RMSE) over a rolling time window (e.g., the last 24 hours of predictions).

•	Prometheus: Deploy Prometheus to your Kubernetes cluster to scrape the /metrics endpoint of your model service periodically. Prometheus collects this time-series data.

•	Grafana: Use Grafana to visualize the data from Prometheus.

---

## 🚀 The MLOps CI/CD is multi-trigger
Code change: CI -> CT ->CD: </br>
Data Change: CT -> CD: New data arrives (e.g., a nightly job aggregates new customer data) </br>
Model Degradation: CT -> CD: </br>


### CI – The Core Trigger Mechanism: The Git Webhook
When a developer (or data scientist) pushes new code to the central repository (e.g., a git push to the main or a feature branch), the following happens:
1.	The Event: The version control system (like GitHub, GitLab, or Azure DevOps) detects the new commit/merge.
2.	The Notification (Webhook): The system immediately sends an automated message, called a webhook, to the central CI/CD orchestrator (e.g. GitHub Actions, Jenkins, or Kubeflow).
3.	The Activation: The orchestrator receives the signal and initiates the first stage of the pipeline (Continuous Integration).

Please check MLOPS\CI-workflow.yml.<br>
This CI workflow is now ready to be saved inside the. github/workflows/ directory of your GitHub repository. 

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]



### It primarily consists of 4 jobs.

### 🛠 Job 1: Unit Testing and Model Performance Check

Runs the test suite across multiple Python versions. Evaluate Model Performance (Precision/Recall Check) <br>
Please check MLOPS\model-performance-pytest.py <br>
        # CRITICAL: python test must use assertions <br>
        # (e.g., 'assert precision >= 0.85') to fail the step if the metric threshold is not met. <br>

In CI-workflow.yml we have below line      </br> 
run: pytest tests/model-performance-pytest.py --cov=src/ --cov-report=xml <br>

--cov=src/ --cov-report=xml <br>
This ensures that while you are running a performance test, you are measuring how much of your source code is being exercised by that test. <br>

<img width="805" height="191" alt="image" src="https://github.com/user-attachments/assets/e38b8246-8192-4aa9-ac61-90108f78ab06" />

run: pytest --cov=./ --cov-report=xml"
The above will run pytest on entire folder.

---

##  📈 Job 2: Build and Push Docker Image
Builds a Docker image, tagging it with the commit SHA and the latest tag (only on pushes to main).<br>
<img width="738" height="232" alt="image" src="https://github.com/user-attachments/assets/2e6cd063-d96e-4116-aa8d-b055cdbd2a52" /> <br>

Please see the MLOPS\Dockerfile for more information.<br>
📌 CI Workflow Context <br>
In CI-workflow.yml, the step that builds the image is:<br>

- name: Build and Push Docker image <br>
        uses: docker/build-push-action@v5 <br>
        with:<br>
          context: .  # This is the build context (the root directory) <br>
          # ... <br>


• The line context: . means the Docker daemon looks in the current directory (the repository root) for the build context. <br>
• By default, the docker build command looks for a file named Dockerfile within that context. <br>
You should place the file directly in the main directory: <br>
Deployment Readiness: Pushes the tested and built image to the GitHub Container Registry (ghcr.io), making it ready for deployment to a service like Kubernetes, Google Cloud Run, or AWS ECS.  <br>

---

## 🕵️ Job 3: Continuous Training (CT) - Train and Store New Model
Please see the continuous_training Job in the CI-workflow.yml file.
This job will use a runner to pull the newly built Docker image, execute the training script inside the container, and handle the resulting model artifact.

docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
#### Run the training script inside the container
          
#### The -v mounts the 'artifacts' directory for the model to be saved locally
  docker run --rm \
     -v ${PWD}/artifacts:/app/artifacts \
     ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest \
     python src/train.py --output-path /app/artifacts/new_model.pkl


## 🤖 Job 4: Deployment to Staging Environment (AWS)
After the CI workflow successfully builds the container image and pushes it to a registry, the CD pipeline is typically triggered by one of three primary mechanisms: 
1.	Image Push Event, 
2.	Git Tagging and Releases (Manual Process), 
3.	Dedicated Deployment Workflow (using GitHub Actions)


#### Implementation using GitHub Actions (Recommended):
We have 2 option to deploy. Either we can use AWS ECS/EKS or we can use AWS sagemaker endpoint.

### 🕵️ Deploy to Staging (AWS ECS/EKS)
Please see the file MLOPS\CI-workflow.yml. This file will go into the repository as (./github/workflows/ CI-workflow.yml): <br>

##### 🔑 Key Deployment Prerequisites
For this Continuous Deployment (CD) job to work, you must set up the following:<br>
•	AWS Infrastructure: You need a running ECS Cluster, an ECS Service, and a Task Definition file (e.g., this should go in github as infrastructure/fraud-detector-task-def.json).<br>
•	GitHub Secrets: <br>
  o	AWS_DEPLOYMENT_ROLE_ARN: You must create an IAM Role in AWS that trusts your GitHub repository via OpenID Connect (OIDC). The ARN for this role must be stored in a GitHub Secret named AWS_OIDC_ROLE_ARN. <br>
  o	AWS_ECS_CLUSTER_NAME: The name of the Amazon ECS cluster where your service runs. <br>
•	Deployment Mechanism: The example uses aws-actions/amazon-ecs-deploy-task-definition@v1 to update an existing ECS service with the new Docker image. <br>

Please see the MLOPS\fraud-detector-task-def.json <br>

### 🕵️ Deploy to SageMaker Endpoint
Please find the CI-workflow-sagemaker.yml for more detail. <br>

🔑 SageMaker Deployment Prerequisites
For this new deployment job to function correctly, you must have the following secrets and infrastructure ready: <br>
1.	Secrets (in GitHub Repository Settings): <br>
o	AWS_DEPLOYMENT_ROLE_ARN: An IAM Role ARN that GitHub Actions assumes to run the deployment commands (must have permissions to create SageMaker Models, Endpoint Configurations, and update Endpoints). <br>
o	AWS_SAGEMAKER_EXECUTION_ROLE_ARN: The SageMaker Execution Role ARN. This is the IAM role that SageMaker itself uses to read data from S3, pull the Docker image, and run the inference containers. <br>
o	AWS_MODEL_BUCKET: The S3 bucket name used for storing the model artifact. <br>
2.	AWS Infrastructure (Pre-existing): <br>
o	A running SageMaker Endpoint named fraud-detection-endpoint (or whatever you set for SAGEMAKER_ENDPOINT_NAME). If the endpoint doesn't exist, the update-endpoint command will fail, and you would need an initial create-endpoint step. <br>
This workflow executes the standard SageMaker deployment pattern: Create Model → Create Endpoint Configuration → Update Endpoint (which performs the actual deployment). <br>

<img width="860" height="649" alt="image" src="https://github.com/user-attachments/assets/df4c66eb-3c96-4a1f-8f79-29b31a50485e" />





