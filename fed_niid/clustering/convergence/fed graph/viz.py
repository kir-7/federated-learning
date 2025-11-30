import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class FedGraphConfig:
    n_clients: int = 10
    m: float = 1.0    # fraction of clients of each cluster that needs to be involved in training
    
    method: str = 'dirichlet'
    dirichlet_alpha: float = 0.3
    dataset: str = "cifar10"    
    n_classes : int = 10
    model : str = 'mnist'
    
    algorithm : str = 'fed_g_prox'
    prox_lambda : float = 0.25
    k_neighbours : float = 0.4
    ema_alpha : float = 0.75 

    n_communities : int = 3

    client_bs: int = 32
    global_bs: int = 32
    
    global_rounds: int = 10
    client_epochs: int = 5
    client_lr: float = 1e-4
    step_size : int = 2
    lr_schd_gamma = 0.5

    train_test_split : float = 0.8
    total_train_samples : int = 100000
    total_test_samples : int = 20000  

    # seed
    torch_seed: int = 42
    np_seed: int = 43
    
    local_eval_every : int  = 3
    start_graph : int = 10
    swap_dist_every : int = 8

    log_dir : str = "checkpoints"
    verbose: bool = True

def plot_training_curves(logger, config, save_path=None):
    """
    Plot training curves from FedGraph logger
    
    Args:
        logger: Dictionary with round_num -> results mapping
        config: FedGraphConfig object
        save_path: Optional path to save the figure
    """
    
    # Extract data from logger
    rounds = sorted(logger.keys())
    
    # Initialize lists
    avg_acc_selected = []
    avg_acc_all = []
    train_losses = []
    
    rounds_with_selected = []
    rounds_with_all = []
    rounds_with_loss = []
    
    for round_num in rounds:
        result = logger[round_num]
        
        # Extract average accuracy for selected clients
        if 'average_acc_selected' in result:
            avg_acc_selected.append(result['average_acc_selected'])
            rounds_with_selected.append(round_num)
        
        # Extract average accuracy for all clients
        if 'average_acc_all' in result:
            avg_acc_all.append(result['average_acc_all'])
            rounds_with_all.append(round_num)
        
        # Extract average training loss
        if 'train_loss' in result and isinstance(result['train_loss'], dict):
            avg_loss = np.mean(list(result['train_loss'].values()))
            train_losses.append(avg_loss)
            rounds_with_loss.append(round_num)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Accuracy curves
    ax1 = axes[0]
    if avg_acc_selected:
        ax1.plot(rounds_with_selected, avg_acc_selected, 
                marker='o', linewidth=2, markersize=4,
                label='Selected Clients Accuracy', color='#2E86AB')
    
    if avg_acc_all:
        ax1.plot(rounds_with_all, avg_acc_all, 
                marker='s', linewidth=2, markersize=4,
                label='All Clients Accuracy', color='#A23B72')
    
    ax1.set_xlabel('Global Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Test Accuracy over Global Rounds\n'
                  f'Algorithm: {config.algorithm} | Dataset: {config.dataset} | '
                  f'Clients: {config.n_clients}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    
    # Plot 2: Training loss
    ax2 = axes[1]
    if train_losses:
        ax2.plot(rounds_with_loss, train_losses, 
                marker='o', linewidth=2, markersize=4,
                label='Average Training Loss', color='#F18F01')
    
    ax2.set_xlabel('Global Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Loss over Global Rounds', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_comparison_curves(loggers_dict, save_path=None):
    """
    Compare multiple experiments on the same plot
    
    Args:
        loggers_dict: Dictionary mapping experiment_name -> logger
        save_path: Optional path to save the figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # Plot accuracy comparison
    ax1 = axes[0]
    for idx, (exp_name, logger) in enumerate(loggers_dict.items()):
        rounds = sorted(logger.keys())
        avg_acc_all = []
        rounds_with_all = []
        
        for round_num in rounds:
            result = logger[round_num]
            if 'average_acc_all' in result:
                avg_acc_all.append(result['average_acc_all'])
                rounds_with_all.append(round_num)
        
        if avg_acc_all:
            ax1.plot(rounds_with_all, avg_acc_all, 
                    marker=markers[idx % len(markers)], 
                    linewidth=2, markersize=5,
                    label=exp_name, 
                    color=colors[idx % len(colors)])
    
    ax1.set_xlabel('Global Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot loss comparison
    ax2 = axes[1]
    for idx, (exp_name, logger) in enumerate(loggers_dict.items()):
        rounds = sorted(logger.keys())
        train_losses = []
        rounds_with_loss = []
        
        for round_num in rounds:
            result = logger[round_num]
            if 'train_loss' in result and isinstance(result['train_loss'], dict):
                avg_loss = np.mean(list(result['train_loss'].values()))
                train_losses.append(avg_loss)
                rounds_with_loss.append(round_num)
        
        if train_losses:
            ax2.plot(rounds_with_loss, train_losses, 
                    marker=markers[idx % len(markers)], 
                    linewidth=2, markersize=5,
                    label=exp_name, 
                    color=colors[idx % len(colors)])
    
    ax2.set_xlabel('Global Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_client_wise_accuracy(logger, config, round_nums=None, save_path=None):
    """
    Plot per-client accuracy at specific rounds
    
    Args:
        logger: Dictionary with round_num -> results mapping
        config: FedGraphConfig object
        round_nums: List of rounds to plot (default: last 3 rounds)
        save_path: Optional path to save the figure
    """
    
    if round_nums is None:
        # Use last 3 rounds by default
        all_rounds = sorted(logger.keys())
        round_nums = all_rounds[-3:] if len(all_rounds) >= 3 else all_rounds
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(config.n_clients)
    width = 0.8 / len(round_nums)
    
    for idx, round_num in enumerate(round_nums):
        # Extract client-wise accuracies (you'll need to add this to your logger)
        # For now, this is a placeholder - you need to log per-client accuracies
        client_accs = []  # TODO: Extract from logger
        
        offset = width * idx - width * (len(round_nums) - 1) / 2
        ax.bar(x + offset, client_accs, width, 
               label=f'Round {round_num}', alpha=0.8)
    
    ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Client Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(config.n_clients)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-client figure saved to {save_path}")
    
    plt.show()
    
    return fig


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Plot after training
    """
    component = FedGraph(config, global_test_data, client_train_partitions, client_test_partitions)
    history = component.run()
    
    # Plot the results
    plot_training_curves(
        component.logger, 
        config, 
        save_path=f"{config.log_dir}/training_curves.png"
    )
    """
    
    # Example 2: Load from saved JSON and plot
    """
    logger, config = load_and_plot_from_json(
        "checkpoints_3/cifar10_15_clients_100_participation_niid_sim_logs_global_eval_round_49.json"
    )
    """
    
    # Example 3: Compare multiple experiments
    """
    loggers = {
        'FedAvg': component_fedavg.logger,
        'FedGProx k=0.3': component_fedgprox_03.logger,
        'FedGProx k=0.6': component_fedgprox_06.logger,
        'FedG-HardCls': component_hardcls.logger,
    }
    
    plot_comparison_curves(
        loggers, 
        save_path=f"{config.log_dir}/comparison.png"
    )
    """
    filepaths_patho = ["patho_cifar10_k_1\patho_cifar10_15_clients_K_1_100_participation_fed_graph.json", \
                 "patho_cifar10_15c\patho_cifar10_15_clients_100_participation_fed_graph.json", \
                 "patho_cifar10_k_0_6\patho_cifar10_15_clients_K_0_6_100_participation_fed_graph.json", \
                 ]
    
    filepaths_rotated = ["dirich_swap_cifar10_k_0_3\dirich_cifar10_15_clients_K_0_33_100_participation_fed_graph.json", \
                         "dirich_noswap_cifar10_k_03\dirich_cifar10_15_clients_K_0_33_100_participation_fed_graph.json", \
                         "dirich_swap_basefed_cifar10_K_0_3\dirich_cifar10_15_clients_K_0_33_100_participation_fed_graph.json"]

    filepaths_mnist_rot = ["dirich_swap_mnist_k_0_3/custom_logs_dirich_rot_mnist_k_0_3.json", \
                           "dirich_mnist_k_0_3\patho_mnist_15_clients_K_0_33_100_participation_fed_graph.json"]

    loggers = {}
    configs = {}
    names = []
    for filepath in filepaths_rotated:
        with open(filepath, 'r') as f:
            d = json.load(f)
            name = "dataset:" + d['config']['dataset'] + " method:" + d['config']['method'] + " algorithm:" + d['config']['algorithm']+ " k=" + str(d['config']["k_neighbours"]) + " swap:" + str(d['config']['swap_dist_every']<d['config']["global_rounds"]) 
            
            d['logs'] = {int(k):v for k, v in d['logs'].items()}

            loggers[name] = d['logs']
            configs[name] = FedGraphConfig(**d['config'])
            names.append(name)
    print(names)
    # plot_comparison_curves(loggers, save_path="./test.png")
    plot_comparison_curves(loggers, save_path='./rot_cifar.png')