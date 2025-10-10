# A Full-Lifecycle Quantum-Enhanced MLOps Pipeline

[![Build Status](https://github.com/satwiksps/quantum-mlops-pipeline/actions/workflows/main.yml/badge.svg)](https://github.com/satwiksps/quantum-mlops-pipeline/actions) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) [![Made with Qiskit](https://img.shields.io/badge/Made%20with-Qiskit-6192FB.svg)](https://qiskit.org/)

This repository contains a novel, end-to-end Quantum MLOps (QMLOps) pipeline that demonstrates a fully automated, self-healing AI system. The project systematically addresses critical bottlenecks in the traditional AI lifecycle—feature engineering, hyperparameter optimization (HPO), and production monitoring—by strategically integrating quantum algorithms.

---
##  Architecture

This project is not a single model, but a complete, automated system composed of three distinct stages. It is designed to be a blueprint for a fully operational, closed-loop MLOps pipeline, where the output of one stage seamlessly becomes the input for the next.


*(**Action Required:** Create a simple diagram showing your three stages and the closed-loop trigger, save it as `pipeline_architecture.png` in a `docs/images` folder, and update this link.)*

---

## Key Features

-   **🧬 Quantum-Native Feature Engineering:** Implements a Hybrid Quantum-Classical Autoencoder (Qiskit + PyTorch) to generate highly expressive features, forming the foundation for all downstream tasks.

-   **⚙️ Quantum-Accelerated HPO:** Automates hyperparameter optimization by mapping the search space to a QUBO problem and solving it with the QAOA algorithm, intelligently navigating the vast search space.

-   **📡 Real-Time Drift Detection:** Deploys a highly sensitive Quantum Support Vector Machine (QSVM) that operates on the quantum-native features to detect subtle data drift in a simulated production stream.

-   **🔄 Autonomous & Self-Healing:** The pipeline features a **closed-loop retraining trigger**. When the Stage 3 monitor detects significant data drift, it automatically initiates a new, independent retraining workflow via the GitHub Actions API.

-   **📊 Full MLOps Instrumentation:** Integrated with **MLflow** for comprehensive experiment tracking of all parameters, metrics, models, and visualizations.

-   **🚀 Robust CI/CD Automation:** Validated by a professional CI/CD workflow using **GitHub Actions** that automatically performs code linting, runs unit tests, and validates the entire pipeline.

-   **🕹️ Centrally Configured:** The entire system is driven by a single, well-documented `config.yaml` file, allowing for easy, code-free experimentation and full reproducibility.

---
## Tech Stack

This project integrates a modern, hybrid tech stack, combining state-of-the-art quantum and classical machine learning libraries.

| Component                  | Technology / Library                                       |
| -------------------------- | ---------------------------------------------------------- |
| **Quantum Computing** | `Qiskit`, `qiskit-aer`, `qiskit-optimization`              |
| **Classical ML & Data** | `PyTorch`, `scikit-learn`, `NumPy`                         |
| **MLOps & Automation** | `MLflow`, `GitHub Actions`                                 |
| **Visualization** | `Matplotlib`                                               |
| **Configuration & Utils** | `PyYAML`, `tqdm`                                           |

---

## Project Structure

The repository is organized into a clean, modular structure to promote maintainability and separation of concerns.

```bash
.
├── .github/workflows/         # Contains all CI/CD automation workflows
│   ├── main.yml               # Main validation pipeline (lint, test, run)
│   └── retrain.yml            # The autonomous retraining pipeline
│
├── saved_models/              # Stores the output model artifacts (ignored by git)
│   ├── feature_extractor/
│   └── tuned_classifier/
│
├── src/                       # The core source code for the project
│   ├── feature_engineering/   # Stage 1: The Hybrid Autoencoder components
│   ├── hyperparameter_tuning/ # Stage 2: The QAOA optimizer components
│   ├── production_monitoring/ # Stage 3: The QSVM drift monitor
│   └── visualization/         # Modular scripts for generating each plot
│
├── tests/                     # Contains all unit tests for the CI pipeline
│
├── config.yaml                # The central "control panel" for all parameters
├── run_pipeline.py            # The main orchestrator script to run the entire pipeline
├── LICENSE                    # The AGPLv3 License file
└── requirements.txt           # A list of all Python dependencies
```
---
## Getting Started

Follow these steps to set up the project environment and run the full MLOps pipeline on your local machine.

### Prerequisites

-   Python (3.9 or higher)
-   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/quantum-enhanced-MLOps.git](https://github.com/your-username/quantum-enhanced-MLOps.git)
    cd quantum-enhanced-MLOps
    ```

2.  **Create and activate a Python virtual environment:**
    * On Windows:
        ```bash
        python -m venv qenv
        .\qenv\Scripts\Activate.ps1
        ```
    * On macOS & Linux:
        ```bash
        python3 -m venv qenv
        source qenv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

The project is designed to be run with two main commands in two separate terminals.

1.  **Execute the MLOps Pipeline:**
    In your first terminal, run the main orchestrator script. This will execute all three stages, save the models, log the experiment to MLflow, and generate the visualizations.
    ```bash
    python run_pipeline.py
    ```

2.  **View the Results in the MLflow UI:**
    After the pipeline has finished, open a **new terminal** (and activate the `qenv` environment). Run the MLflow UI command to start the web server.
    ```bash
    mlflow ui
    ```
    Now, open your web browser and navigate to **`http://127.0.0.1:5000`**. You will see the MLflow dashboard, where you can explore the parameters, metrics, and all saved artifacts from your run.
    
    

### Experiment Configuration

All experiment parameters are controlled from the `config.yaml` file. You can run different experiments without changing any Python code.

-   **For a quick demo (< 2 minutes):** Set `visualization_mode: "fast"`.
-   **For the best quality visuals (can take 15+ minutes):** Set `visualization_mode: "high_quality"`.

---
## Results & Visualizations

The pipeline automatically generates a suite of plots that provide visual proof of the advantages at each stage. The following results were generated using the `high_quality` mode in the `config.yaml` file.

---

### Stage 1: Quantum-Native Feature Space

The Hybrid Autoencoder successfully learns to map the high-dimensional image data into a structured, low-dimensional latent space. As the plot shows, the different digits (represented by colors) are organized into distinct clusters, proving that our quantum layer is creating a powerful and useful representation of the data.


*This demonstrates that the quantum features are meaningful and well-separated.*

`![Feature Space Plot](visualization_stage_1_feature_space.png)`

---

### Stage 2: Quantum-Accelerated Hyperparameter Optimization

This 3D plot visualizes the hyperparameter search space. The scattered teal dots represent a classical random search. The prominent red star marks the single, optimal solution found by our QAOA algorithm, demonstrating the potential of quantum optimization to navigate complex landscapes more effectively.


*This demonstrates that the quantum optimizer intelligently finds the best solution.*

`![HPO Search Plot](visualization_stage_2_hpo_search.png)`

---

### Stage 3: Quantum-Enhanced Production Monitoring

This plot shows our anomaly detector in action. The Quantum SVM has learned a sophisticated, non-linear decision boundary (the line between the red and blue zones) to precisely quarantine the drifted data (orange points) from the normal data (blue points).


*This demonstrates the monitor's sensitivity in detecting subtle data drift.*

`![Drift Detection Plot](visualization_stage_3_drift_boundary.png)`

The performance is quantified in the confusion matrix, which shows a high rate of successful drift detection.

`![Confusion Matrix](visualization_stage_3_confusion_matrix_HIGH_QUALITY.png)`

---
## License

This project is licensed under the **GNU Affero General Public License v3.0**.

See the `LICENSE` file in the root of the repository for the full text and details.