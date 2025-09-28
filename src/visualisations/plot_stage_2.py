# --- Visualization for Stage 2: HPO Search ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.classical_classifier import evaluate_model
from src.hyperparameter_tuning.qubo_formulation import define_hyperparameter_space
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features

colors = {'qaoa_solution': '#E76F51', 'random_search': '#2A9D8F'}

def create_hpo_search_plot():
    """
    Loads artifacts and generates a 3D scatter plot comparing the QAOA solution
    to a classical random search in the hyperparameter space.
    """
    print("\n[VISUALIZATION] Generating plot for Stage 2: HPO Search...")
    
    device = torch.device("cpu")
    latent_dim, img_size = 4, 14
    
    # Load Stage 1 artifacts to generate features needed for evaluation
    encoder = Encoder(latent_dim, img_size)
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    quantum_layer = get_quantum_torch_layer(latent_dim)
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    
    vis_loader, _ = get_data_loaders(batch_size=400, n_samples=400, img_size=img_size)
    features, labels = generate_quantum_features(encoder, quantum_layer, vis_loader, device)

    # We know the QAOA result from the pipeline, so we can define it here
    qaoa_hyperparams = {'hidden_dim': 128, 'lr': 0.001, 'dropout': 0.6}
    qaoa_accuracy = evaluate_model(qaoa_hyperparams, features, labels)
    
    # Simulate a classical random search for comparison
    space = define_hyperparameter_space()
    random_search_results = []
    for _ in range(20):
        hyperparams = {
            'hidden_dim': random.choice(space['hidden_dim']),
            'lr': random.choice(space['lr']),
            'dropout': random.choice(space['dropout'])
        }
        accuracy = evaluate_model(hyperparams, features, labels)
        random_search_results.append((hyperparams, accuracy))

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    dims = [res[0]['hidden_dim'] for res in random_search_results]
    lrs = [res[0]['lr'] for res in random_search_results]
    accs = [res[1] for res in random_search_results]
    ax.scatter(dims, lrs, accs, c=colors['random_search'], s=50, alpha=0.6, label='Classical Random Search')
    ax.scatter(qaoa_hyperparams['hidden_dim'], qaoa_hyperparams['lr'], qaoa_accuracy, c=colors['qaoa_solution'], s=250, marker='*', edgecolor='black', label='QAOA Optimal Solution')
    ax.set_title("QAOA vs. Classical Random Search for HPO", fontsize=16, weight='bold')
    ax.set_xlabel('Hidden Dimension')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel('Model Accuracy')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("visualization_stage_2_hpo_search.png")
    print("--> Saved HPO search plot to visualization_stage_2_hpo_search.png")
    plt.close(fig)

if __name__ == '__main__':
    create_hpo_search_plot()