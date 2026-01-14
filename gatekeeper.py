import mlflow
from mlflow.tracking import MlflowClient
import argparse
import sys

def validate_and_promote(run_id, model_name, metric_name, higher_is_better=True):
    """
    Compares the new training run against the current Production model.
    Registers and promotes if performance is better.
    """
    client = MlflowClient()
    
    # 1. Fetch metrics for the NEW run
    try:
        new_run = client.get_run(run_id)
        new_metric_val = new_run.data.metrics.get(metric_name)
        
        if new_metric_val is None:
            print(f"Error: Metric '{metric_name}' not found in new run {run_id}.")
            sys.exit(1)
            
        print(f"New Run ({run_id}) {metric_name}: {new_metric_val}")
        
    except Exception as e:
        print(f"Failed to fetch new run: {e}")
        sys.exit(1)

    # 2. Fetch the CURRENT Production model
    # We get the latest version currently tagged as "Production"
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    
    promote_new_model = False

    if not prod_versions:
        print(f"No 'Production' model found for {model_name}. This is the first deployment.")
        promote_new_model = True
    else:
        current_prod = prod_versions[0]
        prod_run_id = current_prod.run_id
        
        # Get the metric for the production run
        prod_run = client.get_run(prod_run_id)
        prod_metric_val = prod_run.data.metrics.get(metric_name)
        
        print(f"Current Production ({current_prod.version}) {metric_name}: {prod_metric_val}")

        # 3. Compare Metrics
        if higher_is_better:
            if new_metric_val > prod_metric_val:
                print(f"Pass: {new_metric_val} > {prod_metric_val}")
                promote_new_model = True
        else: # Lower is better (e.g., RMSE, Loss)
            if new_metric_val < prod_metric_val:
                print(f"Pass: {new_metric_val} < {prod_metric_val}")
                promote_new_model = True

    # 4. Register and Transition if validated
    if promote_new_model:
        print("Promoting new model...")
        
        # Register the new model version
        # Note: 'source' depends on where artifacts are stored (S3/AzureBlob/GCS)
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        
        # Transition to Staging (Human approval usually happens between Staging -> Prod)
        # Or transition directly to Production if you trust your automated tests fully
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging" 
        )
        print(f"Success: Model version {mv.version} registered and moved to Staging.")
        
        # Output the new version for GitHub Actions to read
        # GitHub Actions captures this output to pass to the Deployment Job
        print(f"::set-output name=new_version::{mv.version}") 
        
    else:
        print("Fail: New model did not beat production performance.")
        sys.exit(1) # Exit with error to stop the CI pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="MLflow Run ID of the new training")
    parser.add_argument("--model_name", required=True, help="Name of the registered model")
    parser.add_argument("--metric", default="accuracy", help="Metric to compare")
    parser.add_argument("--minimize", action="store_true", help="Set flag if lower metric is better (e.g. RMSE)")
    
    args = parser.parse_args()
    
    validate_and_promote(
        run_id=args.run_id, 
        model_name=args.model_name, 
        metric_name=args.metric, 
        higher_is_better=not args.minimize
    )