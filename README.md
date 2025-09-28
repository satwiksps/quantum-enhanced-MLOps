# A Quantum-Native MLOps Pipeline

This project demonstrates a novel, end-to-end MLOps pipeline that strategically integrates quantum computing at three critical stages: quantum-native feature engineering, quantum-accelerated hyperparameter optimization, and quantum-enhanced production monitoring for data drift.

The entire pipeline is designed to be self-contained and runnable on a standard laptop without a GPU.

## Project Structure

- `run_pipeline.py`: The main script to execute all three MLOps stages sequentially.
- `src/`: Contains the source code for each functional stage.
- `saved_models/`: Stores the outputs (trained models, weights) from each stage.
- `requirements.txt`: A list of all necessary Python packages.

## How to Run

### 1. Setup Environment
It is highly recommended to use a virtual environment.

```bash
python -m venv qenv
.\qenv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
This executes all three stages in order and may take some time.

```bash
python run_pipeline.py
```

## Running Stages Individually
You can also run each stage's script independently. You must run them in order.

```bash
# 1. Run the feature engineering stage
python -m src.feature_engineering.build_feature_extractor

# 2. Run the hyperparameter tuning stage
python -m src.hyperparameter_tuning.tune_with_qaoa

# 3. Run the production monitoring stage
python -m src.production_monitoring.monitor_with_qsvm
```