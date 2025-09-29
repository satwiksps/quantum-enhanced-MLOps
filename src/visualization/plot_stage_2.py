import matplotlib                             # The main plotting library.
matplotlib.use('Agg')                           # Sets the backend to prevent pop-up windows.
import matplotlib.pyplot as plt                 # The primary interface for creating plots.
import torch                                    # The main PyTorch library.
import numpy as np                              # For numerical operations.
import random                                   # For simulating the classical random search.

# --- Local Project Imports ---
from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.classical_classifier import evaluate_model
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features

# --- Defines a consistent color palette for the plot ---
colors = {'qaoa_solution': '#E76F51', 'random_search': '#2A9D8F'}

# The function now accepts the master config object to get its parameters.
def create_hpo_search_plot(config):
    print("\n[VISUALIZATION] Generating plot for Stage 2: HPO Search...")
    
    # --- Read parameters from the config file ---
    cfg1 = config['stage_1_feature_engineering']
    cfg2 = config['stage_2_hyperparameter_tuning']
    device = torch.device("cpu")                # Set the device to CPU.
    
    # --- Load the saved model artifacts from Stage 1 ---
    encoder = Encoder(cfg1['stage_1_latent_dim'], cfg1['stage_1_img_size'])
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth")) # Load encoder weights.
    quantum_layer = get_quantum_torch_layer(cfg1['stage_1_latent_dim'])
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy") # Load PQC weights.
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    
    # --- Generate the feature dataset needed to evaluate different hyperparameters ---
    vis_loader, _ = get_data_loaders(
        batch_size=cfg1['stage_1_n_samples'], 
        n_samples=cfg1['stage_1_n_samples'], 
        img_size=cfg1['stage_1_img_size']
    )
    features, labels = generate_quantum_features(encoder, quantum_layer, vis_loader, device)

    # --- Define the known optimal solution found by the QAOA in the main pipeline ---
    qaoa_hyperparams = {'hidden_dim': 128, 'lr': 0.001, 'dropout': 0.6}
    qaoa_accuracy = evaluate_model(qaoa_hyperparams, features, labels) # Calculate its accuracy score.
    
    # Get the hyperparameter search space from the config file.
    space = cfg2['hyperparameter_space']
    
    # --- Simulate a classical random search to create comparison data points ---
    random_search_results = []
    for _ in range(20):                             # Run 20 random trials.
        hyperparams = {
            'hidden_dim': random.choice(space['hidden_dim']), # Pick a random hidden_dim.
            'lr': random.choice(space['learning_rate']),      # Pick a random learning_rate.
            'dropout': random.choice(space['dropout'])      # Pick a random dropout.
        }
        accuracy = evaluate_model(hyperparams, features, labels) # Evaluate this random combination.
        random_search_results.append((hyperparams, accuracy)) # Store the results.

    # --- Create the 3D Plot ---
    fig = plt.figure(figsize=(12, 9))               # Create a new figure for the plot.
    ax = fig.add_subplot(111, projection='3d')      # Add a 3D subplot to the figure.
    
    # --- Prepare data for plotting ---
    dims = [res[0]['hidden_dim'] for res in random_search_results] # Get all hidden_dim values.
    lrs = [res[0]['lr'] for res in random_search_results]           # Get all learning_rate values.
    accs = [res[1] for res in random_search_results]                # Get all accuracy scores.
    
    # --- Draw the scatter plots ---
    # Plot the random search results as a cloud of teal points.
    ax.scatter(dims, lrs, accs, c=colors['random_search'], s=50, alpha=0.6, label='Classical Random Search')
    # Plot the QAOA solution as a single, large red star.
    ax.scatter(qaoa_hyperparams['hidden_dim'], qaoa_hyperparams['lr'], qaoa_accuracy, c=colors['qaoa_solution'], s=250, marker='*', edgecolor='black', label='QAOA Optimal Solution')
    
    # --- Add Final Touches and Save the Image ---
    ax.set_title("QAOA vs. Classical Random Search for HPO", fontsize=16, weight='bold') # Set the main title.
    ax.set_xlabel('Hidden Dimension')              # Set the x-axis label.
    ax.set_ylabel('Learning Rate')                 # Set the y-axis label.
    ax.set_zlabel('Model Accuracy')                # Set the z-axis label.
    ax.legend()                                    # Display the plot legend.
    
    plt.tight_layout()                             # Adjust plot to prevent labels from overlapping.
    plt.savefig("visualization_stage_2_hpo_search.png") # Save the final plot as a PNG file.
    print("--> Saved HPO search plot to visualization_stage_2_hpo_search.png")
    plt.close(fig)                                 # Close the figure to free up memory.