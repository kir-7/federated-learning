import torchvision
from torchvision import transforms
from scipy.stats import dirichlet


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def distribute_classes_to_clients(n_clients=2, alpha=0.5, n_classes=10):
    """
    First decide how to distribute classes among clients using Dirichlet distribution
    This ensures consistent class distribution between train and val for each client
    
    Args:
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter
        n_classes: Number of classes (10 for MNIST)
    
    Returns:
        class_proportions: Array of shape (n_classes, n_clients) with proportions
    """
    class_proportions = np.zeros((n_classes, n_clients))
    
    # For each class, decide how to distribute it among clients
    for class_idx in range(n_classes):
        # Sample proportions for this class across clients
        proportions = np.random.dirichlet([alpha] * n_clients)
        class_proportions[class_idx] = proportions
    
    return class_proportions

def split_dataset_by_proportions(dataset, class_proportions, max_samples_per_cls=5000):
    """
    Split dataset according to pre-determined class proportions
    
    Args:
        dataset: PyTorch dataset
        class_proportions: Array of shape (n_classes, n_clients)
        max_samples_per_cls: Maximum samples per class to collect
    
    Returns:
        List of client distributions
    """
    n_clients = class_proportions.shape[1]
    cls_to_samples = defaultdict(list)
    
    # Collect all samples grouped by class
    for img, label in dataset:
        if len(cls_to_samples[int(label)]) < max_samples_per_cls:
            cls_to_samples[int(label)].append(img)
    
    client_distributions = [defaultdict(list) for _ in range(n_clients)]
    
    # Distribute samples according to predetermined proportions
    for class_idx in range(10):  # MNIST has 10 classes
        samples = cls_to_samples[class_idx]
        n_samples = len(samples)
        
        if n_samples == 0:
            continue
            
        # Shuffle samples to ensure randomness
        np.random.shuffle(samples)
        
        # Distribute samples according to predetermined proportions
        start_idx = 0
        for client_idx in range(n_clients):
            if client_idx == n_clients - 1:
                # Last client gets all remaining samples
                end_idx = n_samples
            else:
                end_idx = start_idx + int(n_samples * class_proportions[class_idx, client_idx])
            
            client_distributions[client_idx][class_idx] = samples[start_idx:end_idx]
            start_idx = end_idx
    
    return client_distributions

def sample_dirichlet_non_iid(dataset, n_clients=2, alpha=0.5, max_samples_per_cls=1000):
    """
    Sample dataset using Dirichlet distribution for controllable non-IID splits
    
    Args:
        dataset: PyTorch dataset
        n_clients: Number of clients
        alpha: Concentration parameter for Dirichlet distribution
               - alpha >> 1: More IID (uniform distribution)
               - alpha = 1: Uniform Dirichlet
               - alpha < 1: More non-IID (concentrated distribution)
               - alpha → 0: Extremely non-IID
        max_samples_per_cls: Maximum samples per class to collect
    
    Returns:
        List of client distributions
    """
    cls_to_samples = defaultdict(list)
    
    # Collect all samples grouped by class
    for img, label in dataset:
        if len(cls_to_samples[int(label)]) < max_samples_per_cls:
            cls_to_samples[int(label)].append(img)
    
    client_distributions = [defaultdict(list) for _ in range(n_clients)]
    
    # For each class, sample proportions using Dirichlet distribution
    for class_idx in range(10):  # MNIST has 10 classes
        samples = cls_to_samples[class_idx]
        n_samples = len(samples)
        
        if n_samples == 0:
            continue
            
        # Sample proportions for this class across clients
        # Using Dirichlet with concentration parameter alpha
        proportions = np.random.dirichlet([alpha] * n_clients)
        
        # Shuffle samples to ensure randomness
        np.random.shuffle(samples)
        
        # Distribute samples according to sampled proportions
        start_idx = 0
        for client_idx in range(n_clients):
            if client_idx == n_clients - 1:
                # Last client gets all remaining samples
                end_idx = n_samples
            else:
                end_idx = start_idx + int(n_samples * proportions[client_idx])
            
            client_distributions[client_idx][class_idx] = samples[start_idx:end_idx]
            start_idx = end_idx
    
    return client_distributions


def mnist_noniid_sample(dataset_path, n_clients=2, method="dirichlet", 
                       alpha=0.5, concentration=1.0, bias_strength=0.0):
    """
    Main sampling function with multiple methods
    
    Args:
        method: "dirichlet", "concentration", or "lda"
        alpha: Dirichlet/LDA concentration parameter
        concentration: Overall concentration for concentration method
        bias_strength: Bias strength for concentration method
    """
    assert n_clients == 2, "Current implementation supports 2 clients"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(dataset_path, train=True, download=True)
    val_dataset = torchvision.datasets.MNIST(dataset_path, train=False, download=True)
    
    # Choose sampling method
    if method == "dirichlet":
        client_train_dists = sample_dirichlet_non_iid(train_dataset, n_clients, alpha, 1000)
        client_val_dists = sample_dirichlet_non_iid(val_dataset, n_clients, alpha, 100)   
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create global validation set
    global_val_samples = defaultdict(list)
    samples_per_class = 20
    class_counts = defaultdict(int)
    total = 0
    
    for img, label in val_dataset:
        if class_counts[int(label)] < samples_per_class:
            global_val_samples[int(label)].append(img)
            class_counts[int(label)] += 1
            total += 1        
        if total >= 200:
            break
    
    # Format output to match original API
    client_data = [(client_train_dists[i], client_val_dists[i]) for i in range(n_clients)]
    
    return client_data, global_val_samples, transform

def sample_dirichlet_non_iid_consistent(train_dataset, val_dataset, n_clients=2, alpha=0.5, 
                                       max_train_samples=5000, max_val_samples=1000):
    """
    Sample train and validation datasets with consistent class distributions per client
    
    Args:
        train_dataset, val_dataset: PyTorch datasets
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter
        max_train_samples, max_val_samples: Max samples per class for train/val
    
    Returns:
        Tuple of (client_train_distributions, client_val_distributions)
    """
    
    # Step 1: Determine class distribution strategy (same for train and val)
    class_proportions = distribute_classes_to_clients(n_clients, alpha, n_classes=10)
    
    # Step 2: Apply the same proportions to both train and val datasets
    client_train_dists = split_dataset_by_proportions(train_dataset, class_proportions, max_train_samples)
    client_val_dists = split_dataset_by_proportions(val_dataset, class_proportions, max_val_samples)
    
    return client_train_dists, client_val_dists, class_proportions

def mnist_noniid_sample_consistent(dataset_path, n_clients=2, method="dirichlet", 
                                  alpha=0.5):
    from PIL import Image

    """
    Main sampling function with consistent train-val distributions per client
    
    Args:
        method: "dirichlet", "concentration", or "pathological"
        alpha: Dirichlet concentration parameter
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(dataset_path, train=True, download=False)
    val_dataset = torchvision.datasets.MNIST(dataset_path, train=False, download=False)
        
    # Choose sampling method
    if method == "dirichlet":
        client_train_dists, client_val_dists, class_props = sample_dirichlet_non_iid_consistent(
            train_dataset, val_dataset, n_clients, alpha, 1000, 100)    
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create global validation set (balanced across all classes)
    global_val_samples = defaultdict(list)
    samples_per_class = 20
    class_counts = defaultdict(int)
    total = 0
    
    for img, label in val_dataset:
        if class_counts[int(label)] < samples_per_class:
            global_val_samples[int(label)].append(img)
            class_counts[int(label)] += 1
            total += 1        
        if total >= 200:
            break
    
    # Format output to match original API
    client_data = [(client_train_dists[i], client_val_dists[i]) for i in range(n_clients)]
    
    return client_data, global_val_samples, transform, class_props


def plot_client_distributions(client_data, figsize=(15, 10), save_path=None):
    """
    Plot histograms showing sample distributions for all clients' train and validation sets
    
    Args:
        client_data: Output from mnist_noniid_sample_consistent function
                    List of tuples: [(client_1_train, client_1_val), (client_2_train, client_2_val), ...]
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib figure object
    """
    n_clients = len(client_data)
    
    # Create subplots: 2 rows (train/val) x n_clients columns
    fig, axes = plt.subplots(2, n_clients, figsize=figsize)
    
    # Handle case where there's only one client
    if n_clients == 1:
        axes = axes.reshape(2, 1)
    
    # Define colors for consistency
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'khaki', 
              'lightpink', 'lightsalmon', 'lightgray', 'lightblue', 'lightyellow']
    
    # Plot for each client
    for client_idx, (train_dist, val_dist) in enumerate(client_data):
        
        # --- TRAIN DATA PLOT ---
        ax_train = axes[0, client_idx]
        
        # Prepare data for train plot
        train_labels = list(range(10))  # Classes 0-9
        train_counts = [len(train_dist[label]) for label in train_labels]
        
        # Create bar plot for train data
        bars_train = ax_train.bar(train_labels, train_counts, 
                                 color=colors[client_idx % len(colors)], 
                                 alpha=0.7, edgecolor='black', width=0.6)
        
        # Customize train plot
        ax_train.set_title(f'Client {client_idx + 1} - TRAIN', fontweight='bold', fontsize=12)
        ax_train.set_xlabel('Class Labels')
        ax_train.set_ylabel('Number of Samples')
        ax_train.set_xticks(train_labels)
        ax_train.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars_train, train_counts):
            if count > 0:
                ax_train.text(bar.get_x() + bar.get_width()/2, 
                             bar.get_height() + max(train_counts) * 0.01,
                             str(count), ha='center', va='bottom', fontsize=9)
        
        # Set y-axis limit with some padding
        max_train = max(train_counts) if train_counts else 1
        ax_train.set_ylim(0, max_train * 1.1)
        
        # --- VALIDATION DATA PLOT ---
        ax_val = axes[1, client_idx]
        
        # Prepare data for validation plot
        val_labels = list(range(10))  # Classes 0-9
        val_counts = [len(val_dist[label]) for label in val_labels]
        
        # Create bar plot for validation data
        bars_val = ax_val.bar(val_labels, val_counts, 
                             color=colors[client_idx % len(colors)], 
                             alpha=0.7, edgecolor='black', width=0.6)
        
        # Customize validation plot
        ax_val.set_title(f'Client {client_idx + 1} - VALIDATION', fontweight='bold', fontsize=12)
        ax_val.set_xlabel('Class Labels')
        ax_val.set_ylabel('Number of Samples')
        ax_val.set_xticks(val_labels)
        ax_val.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars_val, val_counts):
            if count > 0:
                ax_val.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + max(val_counts) * 0.01,
                           str(count), ha='center', va='bottom', fontsize=9)
        
        # Set y-axis limit with some padding
        max_val = max(val_counts) if val_counts else 1
        ax_val.set_ylim(0, max_val * 1.1)
    
    # Add main title
    fig.suptitle('Client Data Distribution: Train vs Validation', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for main title
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def visualize_client_feature_space(client_data, title=""):
    """
    Visualizes the feature space of client data using t-SNE.

    Args:
        client_data: The output from the sampling function. 
                     A list of tuples, where each tuple is (train_dist, val_dist).
        title (str): The title for the plot.
    """
    all_images_flattened = []
    client_ids = []
    n_clients = len(client_data)

    print("Preparing data for t-SNE...")
    # Loop through each client to gather their data
    for client_id, (train_dist, _) in enumerate(client_data):
        # Loop through each class in the client's training distribution
        for label, images in train_dist.items():
            for img in images:
                # Convert PIL image to a flattened numpy array and normalize
                flat_img = np.array(img).flatten() / 255.0
                all_images_flattened.append(flat_img)
                client_ids.append(client_id)

    X = np.array(all_images_flattened)
    y = np.array(client_ids)

    # Perform t-SNE. It can be slow on large datasets.
    # We can subsample if it's too slow.
    sample_size = min(len(X), 5000) # Limit to 5000 samples for speed
    if len(X) > sample_size:
        print(f"Subsampling data from {len(X)} to {sample_size} points for t-SNE...")
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    print("Running t-SNE... (this may take a moment)")
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(X_sample)

    # Plot the results
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_sample, cmap=plt.get_cmap('jet', n_clients), alpha=0.6)
    
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    
    # Create a legend
    handles, _ = scatter.legend_elements()
    legend_labels = [f"Client {i}" for i in range(n_clients)]
    plt.legend(handles, legend_labels, title="Clients")
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("cifar10_plots/feature_space_viz_noniid.png")

    plt.show()
    

if __name__ == "__main__":

    
    clients_dict, val_samples, *_ = mnist_noniid_sample_consistent("../data/cifar10", 3, alpha=0.3)
    
    # fig = plot_client_distributions(clients_dict, save_path="./cifar10_plots/dirichlet_5_viz.png")
    
    visualize_client_feature_space(clients_dict)