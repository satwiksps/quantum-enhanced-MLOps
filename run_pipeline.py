import os                                       # For interacting with the operating system (e.g., file paths).
import yaml                                     # For reading the main `config.yaml` file.
import mlflow                                   # The main library for experiment tracking.

# --- Import the primary function from each stage's dedicated module ---
from src.feature_engineering.build_feature_extractor import run_feature_engineering
from src.hyperparameter_tuning.tune_with_qaoa import run_hyperparameter_tuning
from src.production_monitoring.monitor_with_qsvm import run_drift_detection
from src.visualization.plot_stage_1 import create_feature_space_plot
from src.visualization.plot_stage_2 import create_hpo_search_plot
from src.visualization.plot_stage_3 import create_drift_detection_plots

def main():
    # --- Step 1: Load the Master Configuration ---
    with open('config.yaml', 'r') as f:             # Open the config file for reading.
        config = yaml.safe_load(f)                  # Load all parameters into a dictionary.

    # --- Step 2: Initialize MLflow Experiment Tracking ---
    mlflow.set_tracking_uri(config['mlflow_tracking_uri']) # Set the folder to store MLflow logs.
    mlflow.set_experiment(config['project_name'])      # Set the experiment name in the MLflow UI.

    # --- Step 3: Start the Master MLflow Run ---
    with mlflow.start_run() as run:                 # Start a new experiment run context.
        run_id = run.info.run_id                    # Get the unique ID for this execution.
        print(f"==========================================================")
        print(f"=== STARTING MLFLOW RUN ID: {run_id} ===")
        print(f"==========================================================")
        
        # --- Flatten all config parameters into a single dictionary for logging ---
        params_to_log = {
            **config['stage_1_feature_engineering'],
            **config['stage_2_hyperparameter_tuning'],
            **config['stage_3_production_monitoring']
        }
        params_to_log.pop('hyperparameter_space', None) # Remove nested dictionaries, as MLflow can't log them.
        
        print("Logging configuration parameters to MLflow...")
        mlflow.log_params(params_to_log)            # Log all parameters to the MLflow run.
        
        # --- Step 4: Execute the MLOps Pipeline Stages Sequentially ---
        run_feature_engineering(config)             # Run Stage 1: Build the feature extractor.
        run_hyperparameter_tuning(config)           # Run Stage 2: Find the best model parameters.
        run_drift_detection(config)                 # Run Stage 3: Monitor for data drift.

        # --- Step 5: Generate Storytelling Visualizations ---
        print("\n==========================================================")
        print("===     MAIN PIPELINE COMPLETE. NOW GENERATING VISUALS...  ===")
        print("==========================================================")
        
        create_feature_space_plot(config)           # Generate the plot for Stage 1.
        create_hpo_search_plot(config)              # Generate the plot for Stage 2.
        create_drift_detection_plots(config)        # Generate the plot(s) for Stage 3.
        
        # --- Step 6: Archive Visualizations in MLflow ---
        print("\nLogging visualization artifacts to MLflow...")
        mlflow.log_artifact("visualization_stage_1_feature_space.png") # Save Stage 1 plot to MLflow.
        mlflow.log_artifact("visualization_stage_2_hpo_search.png") # Save Stage 2 plot to MLflow.
        
        mode = config['stage_3_production_monitoring']['visualization_mode'] # Check which viz mode was used.
        if mode == 'fast':                          # If fast mode was used...
            mlflow.log_artifact("visualization_stage_3_drift_FAST.png") # ...log the fast plot.
            mlflow.log_artifact(f"visualization_stage_3_confusion_matrix_{mode.upper()}.png")
        else:                                       # Otherwise...
            mlflow.log_artifact("visualization_stage_3_drift_boundary.png") # ...log the high-quality plot.
            mlflow.log_artifact(f"visualization_stage_3_confusion_matrix_{mode.upper()}.png")
        
        # --- Final confirmation message ---
        print("\n==========================================================")
        print(f"=== MLOPS PIPELINE EXECUTION COMPLETE FOR RUN ID: {run_id} ===")
        print(f"=== View results in the MLflow UI: `mlflow ui` ===")
        print(f"==========================================================")

# --- Standard Python entry point ---
if __name__ == '__main__':                          # If the script is run directly...
    main()                                          # ...call the main function.