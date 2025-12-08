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
    for (exp_name, exp) in experiments:
        metric_keys.update(exp.get('metrics_distributed_fit', {}).keys())
        metric_keys.update(exp.get('metrics_distributed', {}).keys())
        metric_keys.update(exp.get('metrics_centralized', {}).keys())

    # Exclude specific keys (like your previous code)
    if "similarity_scores" in metric_keys:
        metric_keys.remove("similarity_scores")

    # 3. Determine layout
    # 1 plot for Loss + 1 plot per unique metric
    sorted_keys = sorted(list(metric_keys))
    num_plots = 1 + len(sorted_keys)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))
    if num_plots == 1: axes = [axes] # Handle single plot case

    # Define a color cycle (so each experiment has a unique color)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- PLOT 1: LOSS COMPARISON ---
    ax = axes[0]
    ax.set_title("Loss Evolution (Solid: Centralized, Dashed: Distributed)")
    
    for i, (base_label, exp) in enumerate(experiments):
        config = exp.get('Config', {})
        # base_label = get_exp_label(config)
        color = colors[i % len(colors)] # Cycle colors if > 10 experiments

        # Centralized Loss (Solid Line, Square Marker)
        losses_cent = exp.get('losses_centralized', [])
        if losses_cent:
            # Sort by round to be safe
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

    ax.set_ylabel("Loss")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    # --- PLOT 2+: METRICS COMPARISON ---
    for k, metric_name in enumerate(sorted_keys):
        ax = axes[k + 1]
        ax.set_title(f"{metric_name.capitalize()} Evolution")
        
        for i, (base_label, exp) in enumerate(experiments):
            config = exp.get('Config', {})
            # base_label = get_exp_label(config)
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

        ax.set_ylabel(metric_name)
        ax.set_xlabel("Server Round")
        ax.grid(True, linestyle='--', alpha=0.5)
        # Position legend outside if it's too crowded, or 'best'
        ax.legend() 

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Define your files here
    files_to_compare = [
        "results (client evaluation on val loader)\history_cifar10_fedema_nc_10_m_100_data_drift.pkl",    
        "results (client evaluation on val loader)\history_cifar10_fedknn_nc_10_m_100_data_drift.pkl"    ,
        "results (client evaluation on val loader)\history_cifar10_fedprox_nc_10_m_100_data_drift.pkl"
    ]
    
    plot_compare_experiments(files_to_compare, save_path="./plots/data_drift_100_participation.png")