import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path

def load_data(filepath):
    """Loads JSON or Pickle file into a dictionary."""
    path = Path(filepath)
    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {path}")

def get_exp_label(config):
    """Generates a short label based on Config (Algo + Ratio)."""
    algo = config.get('algorithm', 'Exp')
    ratio = config.get('participation_ratio', '')
    
    label = f"{algo}"
    if ratio:
        label += f" (p={ratio})"
        
    # Optional: Add note if it's short
    # note = config.get('note', '')
    # if note: label += f" - {note[:10]}"
    return label

def plot_compare_experiments(filepaths, save_path=None):
    """
    Plots metrics from multiple experiment files on shared subplots.
    Uses a 2x2 grid layout plus extra plots if needed.
    """
    # 1. Load all experiments
    experiments = []
    for fp in filepaths:
        try:
            data = load_data(fp)
            name = fp.split("\\")[-1]
            experiments.append((name,data))
        except Exception as e:
            print(f"Skipping {fp}: {e}")
    
    if not experiments:
        print("No valid experiments loaded.")
        return

    # 2. Extract unique metric keys from ALL experiments
    metric_keys = set()
    metric_keys.add("accuracy")
    metric_keys.add("avg_train_loss")
    metric_keys.add('recall')
    metric_keys.add('precision')
    metric_keys.add('f1_score') 

    # Exclude specific keys
    if "similarity_scores" in metric_keys:
        metric_keys.remove("similarity_scores")
    
    if "cluster_assignments" in metric_keys:
        metric_keys.remove("cluster_assignments")

    # 3. Determine layout
    # 1 plot for Loss + 1 plot per unique metric
    sorted_keys = sorted(list(metric_keys))
    num_plots = 1 + len(sorted_keys)
    
    # Calculate grid dimensions (2 columns)
    ncols = 2
    nrows = (num_plots + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    
    # Flatten axes array for easier indexing
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Define a color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- PLOT 1: LOSS COMPARISON ---
    ax = axes[0]
    ax.set_title("Loss Evolution (Solid: Centralized, Dashed: Distributed)", fontsize=12, fontweight='bold')
    
    for i, (_, exp) in enumerate(experiments):
        config = exp.get('Config', {})
        base_label = f"exp1_{config['algorithm']}"

        color = colors[i % len(colors)]

        # Centralized Loss (Solid Line, Square Marker)
        losses_cent = exp.get('losses_centralized', [])
        if losses_cent:
            losses_cent.sort(key=lambda x: x[0])
            r, v = zip(*losses_cent)
            ax.plot(r, v, label=f'{base_label} [Cent]', 
                    color=color, linestyle='-', marker='s', markersize=4, alpha=0.8)

        # Distributed Loss (Dashed Line, Circle Marker)
        losses_dist = exp.get('losses_distributed', [])
        if losses_dist:
            losses_dist.sort(key=lambda x: x[0])
            r, v = zip(*losses_dist)
            ax.plot(r, v, label=f'{base_label} [Dist]', 
                    color=color, linestyle='--', marker='o', markersize=4, alpha=0.6)

    ax.set_ylabel("Loss", fontsize=10)
    ax.set_xlabel("Server Round", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    
    # --- PLOT 2+: METRICS COMPARISON ---
    for k, metric_name in enumerate(sorted_keys):
        ax = axes[k + 1]
        ax.set_title(f"{metric_name.capitalize()} Evolution", fontsize=12, fontweight='bold')
        
        for i, (_, exp) in enumerate(experiments):
            config = exp.get('Config', {})
            base_label = f"exp1_{config['algorithm']}"
            color = colors[i % len(colors)]

            # 1. Centralized Eval (Solid, Square)
            data = exp.get('metrics_centralized', {}).get(metric_name, [])
            if data:
                data.sort(key=lambda x: x[0])
                r, v = zip(*data)
                ax.plot(r, v, label=f'{base_label} [Cent]', 
                        color=color, linestyle='-', marker='s', markersize=4)

            # 2. Distributed Eval (Dashed, Circle)
            data = exp.get('metrics_distributed', {}).get(metric_name, [])
            if data:
                data.sort(key=lambda x: x[0])
                r, v = zip(*data)
                ax.plot(r, v, label=f'{base_label} [Dist Eval]', 
                        color=color, linestyle='--', marker='o', markersize=4, alpha=0.6)

            # 3. Distributed Fit (Dotted, Triangle)
            data = exp.get('metrics_distributed_fit', {}).get(metric_name, [])
            if data:
                data.sort(key=lambda x: x[0])
                r, v = zip(*data)
                ax.plot(r, v, label=f'{base_label} [Dist Train]', 
                        color=color, linestyle=':', marker='^', markersize=4, alpha=0.6)

        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_xlabel("Server Round", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Define your files here
    files_to_compare = [
        r"experiments\femnist exps\exp 5 nc 20 m 30 data drift\exp5_femnist_fedavg_nc_20_m_30_data_drift.pkl",
        r"experiments\femnist exps\exp 5 nc 20 m 30 data drift\exp5_femnist_fedknn_nc_20_m_30_data_drift_start_knn_avg.pkl",
        r"experiments\femnist exps\exp 5 nc 20 m 30 data drift\exp5_femnist_fedknn_nc_20_m_30_data_drift_start_knn_sim.pkl",
        r"experiments\femnist exps\exp 5 nc 20 m 30 data drift\exp5_femnist_fedprox_nc_20_m_30_data_drift.pkl",
        r"experiments\femnist exps\exp 5 nc 20 m 30 data drift\exp5_femnist_ifca_nc_20_m_30_data_drift.pkl",
        
        
    ]
    
    plot_compare_experiments(files_to_compare, save_path="plots/experiment5_femnist_noniid_data_dirft_perf.png")