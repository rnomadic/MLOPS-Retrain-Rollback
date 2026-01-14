import argparse
import sys
import mlflow
from mlflow.tracking import MlflowClient

def demote_and_promote(model_name: str):
    """
    Finds the current Production model, archives it, and promotes the
    next best candidate (usually the immediate predecessor) back to Production.
    
    Args:
        model_name: The name of the model in the MLflow Registry.
    """
    client = MlflowClient()
    
    print(f"--- Starting rollback procedure for model: {model_name} ---")

    # 1. Get all versions for the model, sorted by version number (ascending)
    try:
        all_versions = client.get_latest_versions(model_name, stages=['Production', 'Archived', 'Staging', 'None'])
        # Sort by creation time to ensure we identify the history correctly
        all_versions.sort(key=lambda mv: mv.creation_timestamp, reverse=True)
    except Exception as e:
        print(f"Error accessing MLflow Model Registry for {model_name}: {e}")
        sys.exit(1)

    # 2. Identify the currently deployed (bad) Production version
    current_prod_version = next((mv for mv in all_versions if mv.current_stage == 'Production'), None)

    if not current_prod_version:
        print("Warning: No model is currently labeled 'Production'. Registry is already clean.")
        sys.exit(0)

    # 3. Identify the previous good version
    # We assume the last version *before* the current one is the one we want to roll back to.
    
    # Filter out the bad current Production version
    previous_candidates = [mv for mv in all_versions if mv.version != current_prod_version.version]

    # Find the latest model version that is NOT the bad one
    # Note: A more robust check would involve checking deployment history or stored tags.
    if not previous_candidates:
        print("Error: Only one version exists in the registry. Cannot roll back.")
        sys.exit(1)

    # The most recent stable model (e.g., the last one deployed successfully)
    # We will simply promote the most recently created version that isn't the current production one.
    previous_good_version = previous_candidates[0]
    
    print(f"Detected Bad Production Version: v{current_prod_version.version}")
    print(f"Target Good Rollback Version: v{previous_good_version.version}")
    
    # --- Execute Rollback Actions ---
    
    # 4. Demote the bad model (Transition from Production to Archived)
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=current_prod_version.version,
            stage="Archived",
            archive_existing_versions=False # We handle the promotion next
        )
        print(f"Success: Demoted v{current_prod_version.version} to 'Archived'.")
    except Exception as e:
        print(f"Error archiving version v{current_prod_version.version}: {e}")
        # Continue attempting promotion if demotion fails, but log the error
        
    # 5. Promote the previous good model back to Production
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=previous_good_version.version,
            stage="Production",
            archive_existing_versions=False
        )
        print(f"Success: Promoted v{previous_good_version.version} back to 'Production'.")
    except Exception as e:
        print(f"Fatal Error: Failed to promote v{previous_good_version.version} back to Production: {e}")
        sys.exit(1)

    print("--- MLflow Model Registry rollback complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automates MLflow Model Registry rollback.")
    parser.add_argument("--model_name", required=True, help="The registered model name (e.g., 'fraud-detection-model').")
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI from environment variable
    if "MLFLOW_TRACKING_URI" not in os.environ:
        print("FATAL: Environment variable MLFLOW_TRACKING_URI is not set.")
        sys.exit(1)
        
    demote_and_promote(args.model_name)

