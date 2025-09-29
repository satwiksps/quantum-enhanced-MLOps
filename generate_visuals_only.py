# ===================================================================
# Standalone Script to Generate All Visualizations
# ===================================================================
# Use this script to regenerate plots without re-running the entire
# MLOps pipeline. It requires that `run_pipeline.py` has been
# successfully run at least once to create the necessary saved models.

import yaml

# Import the individual, modular plotting scripts
from src.visualization.plot_stage_1 import create_feature_space_plot
from src.visualization.plot_stage_2 import create_hpo_search_plot
from src.visualization.plot_stage_3 import create_drift_detection_plots

def main():
    """
    Loads the project configuration and runs all visualization functions.
    """
    
    # 1. Load the project configuration from the YAML file
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("[ERROR] `config.yaml` not found! Please ensure it's in the root directory.")
        return
        
    print("==========================================================")
    print("===      RE-GENERATING ALL STORYTELLING VISUALS...       ===")
    print("==========================================================")
    
    # 2. Call each visualization function, passing the config object
    #    so they know which settings to use (e.g., 'fast' vs 'high_quality').
    create_feature_space_plot(config)
    create_hpo_search_plot(config)
    create_drift_detection_plots(config)
    
    print("\n==========================================================")
    print("===         ALL VISUALS RE-GENERATED SUCCESSFULLY        ===")
    print("==========================================================")

if __name__ == '__main__':
    main()