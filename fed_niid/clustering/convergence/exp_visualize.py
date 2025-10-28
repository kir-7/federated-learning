import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Import your FedConverge class file
# from your_script_name import FedConverge, FedConvergeConfig

class FedVisualizer:
    """
    A class to visualize the convergence and client drift dynamics
    from a trained FedConverge experiment, treating each client individually.
    """
    def __init__(self, fed_experiment):
        """
        Initializes the visualizer with a completed FedConverge experiment object.
        
        Args:
            fed_experiment: An instance of your FedConverge class after .run() has been called.
        """
        self.fed_experiment = fed_experiment
        self.config = fed_experiment.config
        self.coords_df = None
        print("Visualizer initialized. Call .prepare_data_for_plotting() to proceed.")

    def _flatten_model(self, model):
        """Flattens a PyTorch model's parameters into a single NumPy vector."""
        # Ensure model is on CPU before converting to numpy
        return np.concatenate([
            p.detach().cpu().numpy().flatten() 
            for p in model.parameters()
        ])

    def _get_client_data_distributions(self):
        """Calculates the class distribution for each client's training data."""
        print("Calculating client data distributions...")
        distributions = []
        for client_id in tqdm(range(self.config.n_clients)):
            loader = self.fed_experiment.client_loaders_train[client_id]
            counts = np.zeros(self.config.n_classes, dtype=int)
            for batch in loader:
                labels = batch['label']
                for i in range(self.config.n_classes):
                    counts[i] += (labels == i).sum().item()
            total = counts.sum()
            distributions.append(counts / total if total > 0 else counts)
        return np.array(distributions)

    def prepare_data_for_plotting(self, dim_reduce_method='pca'):
        """
        Performs the heavy lifting:
        1. Collects and flattens all model states.
        2. Applies dimensionality reduction (PCA or t-SNE).
        3. Stores the results in a convenient DataFrame.
        """
        # 1. Collect and flatten all models
        print(f"Collecting and flattening all {len(self.fed_experiment.gloabl_model_states)} global and local models...")
        all_models_flat = []
        metadata = []

        # Global models
        for round_num, model in self.fed_experiment.gloabl_model_states.items():
            all_models_flat.append(self._flatten_model(model))
            metadata.append({'round': round_num, 'client_id': 'Global', 'type': 'global'})
            
        # Local models
        for client_id, round_dict in self.fed_experiment.local_model_states.items():
            for round_num, model in round_dict.items():
                if round_num > 0:
                    all_models_flat.append(self._flatten_model(model))
                    metadata.append({'round': round_num, 'client_id': client_id, 'type': 'local'})

        all_models_flat = np.array(all_models_flat)
        
        # 2. Apply dimensionality reduction
        print(f"Applying {dim_reduce_method.upper()} to {all_models_flat.shape[0]} model vectors...")
        if dim_reduce_method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=self.config.np_seed)
        elif dim_reduce_method.lower() == 'tsne':
            perplexity = min(30, all_models_flat.shape[0] - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=self.config.np_seed)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
            
        coords = reducer.fit_transform(all_models_flat)
        
        # 3. Store in DataFrame
        self.coords_df = pd.DataFrame(metadata)
        self.coords_df[['x', 'y']] = coords
        
        print("Data preparation complete. You can now use the plotting functions.")

    def plot_data_distribution(self):
        """Plots a stacked bar chart of the data distribution for each client."""
        distributions = self._get_client_data_distributions()
        
        df_dist = pd.DataFrame(distributions, columns=[f'Class {i}' for i in range(self.config.n_classes)])
        df_dist['client_id'] = df_dist.index
        
        df_dist.set_index('client_id').plot(
            kind='bar', 
            stacked=True, 
            figsize=(15, 7),
            cmap='tab20',
            width=0.8
        )
        
        plt.title('Client Data Distributions', fontsize=16)
        plt.xlabel('Client ID', fontsize=12)
        plt.ylabel('Proportion of Data', fontsize=12)
        plt.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_round_snapshot(self, round_num):
        """
        Creates a scatter plot showing model positions for a single round,
        coloring each client individually.
        """
        if self.coords_df is None:
            print("Please run .prepare_data_for_plotting() first.")
            return
            
        plt.figure(figsize=(12, 10))
        
        df_round = self.coords_df[self.coords_df['round'] == round_num]
        start_global = self.coords_df[(self.coords_df['round'] == round_num - 1) & (self.coords_df['type'] == 'global')]
        end_global = self.coords_df[(self.coords_df['round'] == round_num) & (self.coords_df['type'] == 'global')]
        
        if start_global.empty or end_global.empty:
            print(f"Data for round {round_num} or {round_num-1} not found.")
            return

        start_pos = start_global[['x', 'y']].iloc[0]
        
        # Disable legend if there are too many clients
        legend_option = 'full' if self.config.n_clients <= 20 else False

        # Plot client models
        sns.scatterplot(
            data=df_round[df_round['type'] == 'local'],
            x='x', y='y',
            hue='client_id',
            palette='tab20', # Good for up to 20 unique colors
            s=100,
            alpha=0.9,
            legend=legend_option
        )
        
        # Plot start/end global models
        plt.scatter(start_pos['x'], start_pos['y'], marker='x', color='gray', s=250, label='Start Global Model', zorder=5)
        plt.scatter(end_global['x'], end_global['y'], marker='*', color='black', s=450, label='End Global (Aggregated)', zorder=5)

        # Draw drift vectors
        for _, row in df_round[df_round['type'] == 'local'].iterrows():
            plt.arrow(start_pos['x'], start_pos['y'], row['x'] - start_pos['x'], row['y'] - start_pos['y'],
                      alpha=0.2, color='gray', head_width=0)

        plt.title(f'Model Space Snapshot - Round {round_num}', fontsize=16)
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.legend(title='Client ID / Model')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
    def plot_trajectories(self):
        """Plots the full trajectories, coloring each client individually."""
        if self.coords_df is None:
            print("Please run .prepare_data_for_plotting() first.")
            return

        plt.figure(figsize=(12, 10))
        
        df_local = self.coords_df[self.coords_df['type'] == 'local'].copy()
        df_local['client_id'] = df_local['client_id'].astype(int) # Ensure client_id is numeric for plotting
        
        df_global = self.coords_df[self.coords_df['type'] == 'global']

        # Disable legend if there are too many clients
        legend_option = 'full' if self.config.n_clients <= 20 else False

        # Plot local model trajectories
        sns.lineplot(
            data=df_local,
            x='x', y='y',
            hue='client_id',
            units='client_id',
            estimator=None,
            palette='tab20',
            alpha=0.5,
            legend=legend_option
        )
        
        # Plot global model trajectory
        plt.plot(df_global['x'], df_global['y'], 'k-o', markersize=8, linewidth=3, label='Global Model Path', zorder=10)
        plt.scatter(df_global.iloc[0]['x'], df_global.iloc[0]['y'], marker='s', s=200, color='lime', label='Start', zorder=11, edgecolors='black')
        plt.scatter(df_global.iloc[-1]['x'], df_global.iloc[-1]['y'], marker='*', s=400, color='gold', label='Finish', zorder=11, edgecolors='black')

        plt.title('Model Trajectories Across All Rounds', fontsize=16)
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        if legend_option:
             plt.legend(title='Client ID / Model', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def create_animation(self, filename='drift_animation.gif'):
        """Creates and saves a GIF animation of the training process."""
        if self.coords_df is None:
            print("Please run .prepare_data_for_plotting() first.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        
        x_min, x_max = self.coords_df['x'].min(), self.coords_df['x'].max()
        y_min, y_max = self.coords_df['y'].min(), self.coords_df['y'].max()
        x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        legend_option = False if self.config.n_clients > 10 else 'full' # smaller threshold for animation frames

        def update(round_num):
            round_num += 1
            ax.clear()
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            
            df_round = self.coords_df[self.coords_df['round'] == round_num]
            start_global = self.coords_df[(self.coords_df['round'] == round_num - 1) & (self.coords_df['type'] == 'global')]
            end_global = self.coords_df[(self.coords_df['round'] == round_num) & (self.coords_df['type'] == 'global')]
            
            if start_global.empty or end_global.empty: return

            start_pos = start_global[['x', 'y']].iloc[0]

            sns.scatterplot(data=df_round[df_round['type'] == 'local'], x='x', y='y', hue='client_id', palette='tab20', s=100, alpha=0.8, ax=ax, legend=legend_option)
            ax.scatter(start_pos['x'], start_pos['y'], marker='x', color='gray', s=200, label='Start Global')
            ax.scatter(end_global['x'], end_global['y'], marker='*', color='black', s=400, label='End Global')

            ax.set_title(f'Model Space - Round {round_num}', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.5)
            if legend_option: ax.legend(loc='upper right')

        ani = animation.FuncAnimation(fig, update, frames=self.config.global_rounds, interval=500, repeat=False)
        
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow')
        print("Animation saved.")
        plt.close()