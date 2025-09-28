import os
from src.feature_engineering.build_feature_extractor import run_feature_engineering
from src.hyperparameter_tuning.tune_with_qaoa import run_hyperparameter_tuning
from src.production_monitoring.monitor_with_qsvm import run_drift_detection

def main():
    """Executes the entire Quantum-Native MLOps pipeline from end to end."""
    print("==========================================================")
    print("=== EXECUTING END-TO-END QUANTUM-NATIVE MLOPS PIPELINE ===")
    print("==========================================================")
    
    # --- Stage 1: Feature Engineering ---
    p1_params = {
        'latent_dim': 4, 'epochs': 3, 'lr': 0.005,
        'batch_size': 16, 'n_samples': 200, 'img_size': 14
    }
    run_feature_engineering(**p1_params)
    
    stage1_output_exists = os.path.exists("saved_models/feature_extractor/hae_encoder.pth")
    if not stage1_output_exists:
        print("\n[ERROR] Stage 1 did not produce model files. Aborting pipeline.")
        return

    # --- Stage 2: Hyperparameter Tuning ---
    run_hyperparameter_tuning()

    # --- Stage 3: Production Monitoring ---
    run_drift_detection()

    print("\n==========================================================")
    print("===         MLOPS PIPELINE EXECUTION COMPLETE          ===")
    print("==========================================================")

if __name__ == '__main__':
    main()