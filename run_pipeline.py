import os
import yaml
import mlflow

from src.feature_engineering.build_feature_extractor import run_feature_engineering
from src.hyperparameter_tuning.tune_with_qaoa import run_hyperparameter_tuning
from src.production_monitoring.monitor_with_qsvm import run_drift_detection
from src.visualization.plot_stage_1 import create_feature_space_plot
from src.visualization.plot_stage_2 import create_hpo_search_plot
from src.visualization.plot_stage_3 import create_drift_detection_plots

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment(config['project_name'])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"==========================================================")
        print(f"=== STARTING MLFLOW RUN ID: {run_id} ===")
        print(f"==========================================================")
        
        # --- THE FIX: Create a single dictionary with unique keys before logging ---
        params_to_log = {
            **config['stage_1_feature_engineering'],
            **config['stage_2_hyperparameter_tuning'],
            **config['stage_3_production_monitoring']
        }
        # MLflow cannot log nested dictionaries, so we remove this one.
        params_to_log.pop('hyperparameter_space', None) 
        
        print("Logging configuration parameters to MLflow...")
        mlflow.log_params(params_to_log)
        
        run_feature_engineering(config)
        run_hyperparameter_tuning(config)
        run_drift_detection(config)

        print("\n==========================================================")
        print("===     MAIN PIPELINE COMPLETE. NOW GENERATING VISUALS...  ===")
        print("==========================================================")
        
        create_feature_space_plot(config)
        create_hpo_search_plot(config)
        create_drift_detection_plots(config)
        
        print("\nLogging visualization artifacts to MLflow...")
        mlflow.log_artifact("visualization_stage_1_feature_space.png")
        mlflow.log_artifact("visualization_stage_2_hpo_search.png")
        mode = config['stage_3_production_monitoring']['visualization_mode']
        if mode == 'fast':
            mlflow.log_artifact("visualization_stage_3_drift_FAST.png")
            mlflow.log_artifact(f"visualization_stage_3_confusion_matrix_{mode.upper()}.png")
        else:
            mlflow.log_artifact("visualization_stage_3_drift_boundary.png")
            mlflow.log_artifact(f"visualization_stage_3_confusion_matrix_{mode.upper()}.png")
        
        print("\n==========================================================")
        print(f"=== MLOPS PIPELINE EXECUTION COMPLETE FOR RUN ID: {run_id} ===")
        print(f"=== View results in the MLflow UI: `mlflow ui` ===")
        print(f"==========================================================")

if __name__ == '__main__':
    main()