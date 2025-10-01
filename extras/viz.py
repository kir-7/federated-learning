import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from matplotlib.lines import Line2D

class FederatedLearningVisualizer:
    def __init__(self):
        self.client_weights_history = {}  # {round: {client_id: state_dict}}
        self.global_weights_history = {}  # {round: state_dict}

    def add_global_weights(self, round_num, global_weights_dict):
        """
        Add the global (averaged) weights for a specific round.
        The initial model should be added as round 0.
        
        Args:
            round_num: The global round number (use 0 for the initial model)
            global_weights_dict: The model.state_dict() of the global model
        """
        self.global_weights_history[round_num] = global_weights_dict
    def add_round_weights(self, round_num, client_weights_dict):
        """
        Add weights for all clients in a specific round
        
        Args:
            round_num: The global round number
            client_weights_dict: Dictionary {client_id: model.state_dict()}
        """
        self.client_weights_history[round_num] = client_weights_dict
    
    def flatten_state_dict(self, state_dict):
        """Flatten all parameters in state_dict to a 1D vector"""
        flattened = []
        for param in state_dict.values():
            flattened.extend(param.flatten().cpu().numpy())
        return np.array(flattened)
    
    def compute_pairwise_distances(self, round_num):
        """Compute pairwise distances between all clients in a round"""
        if round_num not in self.client_weights_history:
            raise ValueError(f"Round {round_num} not found")
        
        client_weights = self.client_weights_history[round_num]
        client_ids = list(client_weights.keys())
        n_clients = len(client_ids)
        
        # Flatten all client weights
        flattened_weights = []
        for client_id in client_ids:
            flattened = self.flatten_state_dict(client_weights[client_id])
            flattened_weights.append(flattened)
        
        # Compute pairwise distances
        distances = pdist(flattened_weights, metric='euclidean')
        distance_matrix = squareform(distances)
        
        return distance_matrix, client_ids
    
    def plot_convergence_heatmap(self, figsize=(12, 8)):
        """Plot heatmap showing pairwise distances between clients over rounds"""
        rounds = sorted(self.client_weights_history.keys())
        n_rounds = len(rounds)
        
        fig, axes = plt.subplots(2, (n_rounds + 1) // 2, figsize=figsize)
        if n_rounds == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for i, round_num in enumerate(rounds):
            distance_matrix, client_ids = self.compute_pairwise_distances(round_num)
            
            sns.heatmap(distance_matrix, 
                       xticklabels=client_ids, 
                       yticklabels=client_ids,
                       annot=True, 
                       fmt='.2f',
                       cmap='viridis_r',
                       ax=axes[i])
            axes[i].set_title(f'Round {round_num} - Client Weight Distances')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    
    def plot_weight_evolution_pca(self, n_components=2):
        """
        Visualize the evolution of client and global weights using PCA,
        annotating each point with its round number.

        This plot shows:
        - Global models as large stars.
        - Client models as smaller circles.
        - Each point is numerically labeled with its corresponding federated round.
        - An inset heatmap in the corner shows the similarity matrix between clients,
          if `self.similarity_matrix` is available.
        """
        if n_components != 2:
            print("This visualization is designed for 2 components.")
            n_components = 2

        # --- Data Collection (same as before) ---
        all_weights = []
        labels = []

        global_rounds = sorted(self.global_weights_history.keys())
        for r in global_rounds:
            state_dict = self.global_weights_history[r]
            all_weights.append(self.flatten_state_dict(state_dict))
            labels.append({'round': r, 'type': 'global', 'id': 'global'})

        client_rounds = sorted(self.client_weights_history.keys())
        for r in client_rounds:
            for client_id, state_dict in self.client_weights_history[r].items():
                all_weights.append(self.flatten_state_dict(state_dict))
                labels.append({'round': r, 'type': 'client', 'id': client_id})
                
        if not all_weights:
            print("No weights to visualize. Add weights using add_round_weights() and add_global_weights().")
            return plt.figure()
            
        all_weights = np.array(all_weights)
        
        # --- Dimensionality Reduction (same as before) ---
        n_samples = len(all_weights)
        
        if n_samples <= 1:
            print("PCA requires at least 2 data points.")
            return plt.figure()
            
        pca = PCA(n_components=n_components)
        projected_weights = pca.fit_transform(all_weights)

        # --- Data Organization (same as before) ---
        projected_global = {}
        projected_local = {}
        for i, label in enumerate(labels):
            r = label['round']
            if label['type'] == 'global':
                projected_global[r] = projected_weights[i]
            else:
                if r not in projected_local:
                    projected_local[r] = {}
                projected_local[r][label['id']] = projected_weights[i]

        # --- PLOTTING LOGIC ---
        fig, ax = plt.subplots(figsize=(16, 10))
        
        all_rounds = sorted(list(set(global_rounds + client_rounds)))
        # Handle case with no rounds
        if not all_rounds:
            print("No rounds found to plot.")
            return fig
            
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_rounds) + 1))
        round_color_map = {r: colors[i] for i, r in enumerate(all_rounds)}

        # Plot global model points and add text labels
        for r, pos in sorted(projected_global.items()):
            ax.scatter(pos[0], pos[1], c=[round_color_map.get(r, 'k')], marker='*', s=600,
                    zorder=4, edgecolors='k', lw=1.5)
            ax.text(pos[0], pos[1], str(r), color='white', ha='center', va='center',
                    fontweight='bold', fontsize=12, zorder=5)

        # Plot client model points and add text labels
        for r, clients in sorted(projected_local.items()):
            for client_id, pos in clients.items():
                ax.scatter(pos[0], pos[1], c=[round_color_map[r]], marker='o', s=150,
                        alpha=0.8, zorder=3, edgecolors='k', lw=0.5)
                ax.text(pos[0], pos[1], str(r), color='black', ha='center', va='center',
                        fontweight='medium', fontsize=9, zorder=5)

        # --- Aesthetics and Updated Legend ---
        ax.set_title('PCA Visualization of Federated Learning Weight Evolution', fontsize=16)
        ax.set_xlabel('PCA Component 1', fontsize=12)
        ax.set_ylabel('PCA Component 2', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        legend_elements = [
            Line2D([0], [0], marker='*', color='grey', label='Global Model (Averaged)', 
                markersize=20, ls='None', markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='grey', label='Client Model (Local Update)', 
                markersize=12, ls='None', markeredgecolor='k')
        ]
        ax.legend(handles=legend_elements, fontsize=12, loc='upper left')

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(all_rounds), vmax=max(all_rounds)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, ticks=all_rounds)
        cbar.set_label('Federated Round', fontsize=12)
        
        # --- NEW: ADD SIMILARITY MATRIX INSET ---
        if hasattr(self, 'similarity_matrix') and self.similarity_matrix is not None:
            # Position the inset axes in the upper right corner
            # [left, bottom, width, height] in figure coordinates
            ax_inset = fig.add_axes([0.68, 0.65, 0.2, 0.2])
            
            im = ax_inset.imshow(self.similarity_matrix, cmap='cividis', interpolation='nearest')
            
            # Add client ID labels as ticks
            if hasattr(self, 'client_ids') and self.client_ids is not None:
                tick_labels = self.client_ids
            else: # Fallback to generic labels
                n_clients = self.similarity_matrix.shape[0]
                tick_labels = [f'C{i}' for i in range(n_clients)]

            ax_inset.set_xticks(np.arange(len(tick_labels)))
            ax_inset.set_yticks(np.arange(len(tick_labels)))
            ax_inset.set_xticklabels(tick_labels, fontsize=8)
            ax_inset.set_yticklabels(tick_labels, fontsize=8)

            # Rotate the x-axis labels to prevent overlap
            plt.setp(ax_inset.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add a colorbar for the inset plot
            cbar_inset = fig.colorbar(im, ax=ax_inset, fraction=0.046, pad=0.04)
            cbar_inset.ax.tick_params(labelsize=8)

            ax_inset.set_title('Client Data Similarity', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to prevent colorbar overlap
        return fig