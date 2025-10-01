import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def visualize_clustering(fedcluster_instances, method='both', figsize=(20, 12), 
                        fixed_scale=True, instance_names=None):
    """
    Visualize the clustering results using multiple techniques with consistent scaling
    
    Parameters:
    - fedcluster_instances: Single FedCluster object or list of FedCluster objects
    - method: 'pca', 'tsne', or 'both' for dimensionality reduction
    - figsize: Figure size for the plots
    - fixed_scale: If True, use consistent scales across all instances
    - instance_names: List of names for each instance (e.g., ['IID', 'Non-IID'])
    """
    
    # Convert single instance to list for uniform handling
    if not isinstance(fedcluster_instances, list):
        fedcluster_instances = [fedcluster_instances]
        instance_names = instance_names or ['Single Instance']
    else:
        instance_names = instance_names or [f'Instance {i+1}' for i in range(len(fedcluster_instances))]
    
    n_instances = len(fedcluster_instances)
    
    # Prepare data for all instances
    all_feature_matrices = []
    all_cluster_labels = []
    all_n_clients = []
    
    for instance in fedcluster_instances:
        distribution_stats = instance.get_distribution_stats()
        feature_matrix = instance.create_feature_matrix(distribution_stats)
        all_feature_matrices.append(feature_matrix)
        all_cluster_labels.append(instance.labels_)
        all_n_clients.append(instance.n_clients)
    
    # Global scaling parameters if fixed_scale is True
    if fixed_scale and len(all_feature_matrices) > 1:
        # Combine all feature matrices to get global min/max
        combined_features = np.vstack(all_feature_matrices)
        global_feature_range = (combined_features.min(), combined_features.max())
        
        # Fit PCA and t-SNE on combined data for consistent transformation
        global_pca = PCA(n_components=2)
        global_pca.fit(combined_features)
        
        if combined_features.shape[0] > 3:
            global_tsne = TSNE(n_components=2, random_state=42, 
                              perplexity=min(3, combined_features.shape[0]-1))
            global_tsne_coords = global_tsne.fit_transform(combined_features)
            
            # Calculate global coordinate ranges
            global_pca_coords = global_pca.transform(combined_features)
            pca_x_range = (global_pca_coords[:, 0].min(), global_pca_coords[:, 0].max())
            pca_y_range = (global_pca_coords[:, 1].min(), global_pca_coords[:, 1].max())
            tsne_x_range = (global_tsne_coords[:, 0].min(), global_tsne_coords[:, 0].max())
            tsne_y_range = (global_tsne_coords[:, 1].min(), global_tsne_coords[:, 1].max())
    
    # Determine subplot layout
    if method == 'both':
        n_cols = 3
        n_rows = 2 * n_instances
    else:
        n_cols = 3
        n_rows = n_instances
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Federated Learning Clustering Comparison', fontsize=16, fontweight='bold')
    
    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each instance
    for inst_idx, (instance, feature_matrix, cluster_labels, n_clients) in enumerate(
        zip(fedcluster_instances, all_feature_matrices, all_cluster_labels, all_n_clients)):
        
        client_ids = np.arange(n_clients)
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        # Calculate row indices for this instance
        if method == 'both':
            pca_row = inst_idx * 2
            tsne_row = inst_idx * 2 + 1
        else:
            pca_row = tsne_row = inst_idx
        
        # PCA Visualization
        if method in ['pca', 'both']:
            if fixed_scale and len(all_feature_matrices) > 1:
                pca_coords = global_pca.transform(feature_matrix)
            else:
                pca = PCA(n_components=2)
                pca_coords = pca.fit_transform(feature_matrix)
            
            ax = axes[pca_row, 0]
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = cluster_labels == cluster_id
                ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                          c=[colors[i]], label=f'Cluster {cluster_id}', 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1)
                
                # Add client ID annotations
                for j, client_id in enumerate(client_ids[mask]):
                    ax.annotate(f'C{client_id}', 
                               (pca_coords[mask][j, 0], pca_coords[mask][j, 1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Set consistent scales if requested
            if fixed_scale and len(all_feature_matrices) > 1:
                ax.set_xlim(pca_x_range[0] - 0.1 * (pca_x_range[1] - pca_x_range[0]), 
                           pca_x_range[1] + 0.1 * (pca_x_range[1] - pca_x_range[0]))
                ax.set_ylim(pca_y_range[0] - 0.1 * (pca_y_range[1] - pca_y_range[0]), 
                           pca_y_range[1] + 0.1 * (pca_y_range[1] - pca_y_range[0]))
                ax.set_xlabel(f'PC1 ({global_pca.explained_variance_ratio_[0]:.2%} variance)')
                ax.set_ylabel(f'PC2 ({global_pca.explained_variance_ratio_[1]:.2%} variance)')
            else:
                if not fixed_scale:
                    pca = PCA(n_components=2)
                    pca.fit(feature_matrix)
                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            
            ax.set_title(f'PCA: {instance_names[inst_idx]} Clustering')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Class distribution heatmap
            ax_heatmap = axes[pca_row, 1]
            
            df_dist = pd.DataFrame(feature_matrix, 
                                  columns=[f'Class_{i}' for i in range(feature_matrix.shape[1])],
                                  index=[f'Client_{i}' for i in range(n_clients)])
            df_dist['Cluster'] = cluster_labels
            df_dist = df_dist.sort_values('Cluster')
            
            # Use consistent color scale for heatmaps
            if fixed_scale and len(all_feature_matrices) > 1:
                vmin, vmax = global_feature_range
            else:
                vmin, vmax = None, None
            
            sns.heatmap(df_dist.iloc[:, :-1], annot=True, cmap='Blues', 
                       fmt='.3f', ax=ax_heatmap, cbar_kws={'label': 'Class Proportion'},
                       vmin=vmin, vmax=vmax)
            ax_heatmap.set_title(f'{instance_names[inst_idx]} Class Distribution')
            ax_heatmap.set_ylabel('Clients (sorted by cluster)')
            
            # Add cluster boundaries
            cluster_boundaries = []
            current_cluster = df_dist['Cluster'].iloc[0]
            for i, cluster in enumerate(df_dist['Cluster']):
                if cluster != current_cluster:
                    cluster_boundaries.append(i)
                    current_cluster = cluster
            
            for boundary in cluster_boundaries:
                ax_heatmap.axhline(y=boundary, color='red', linewidth=2)
            
            # Cluster statistics
            ax_stats = axes[pca_row, 2]
            
            cluster_sizes = [np.sum(cluster_labels == i) for i in unique_clusters]
            bars = ax_stats.bar(unique_clusters, cluster_sizes, color=colors, alpha=0.7, edgecolor='black')
            ax_stats.set_xlabel('Cluster ID')
            ax_stats.set_ylabel('Number of Clients')
            ax_stats.set_title(f'{instance_names[inst_idx]} Clients per Cluster')
            ax_stats.grid(True, alpha=0.3)
            
            # Set consistent y-axis scale for cluster size plots
            if fixed_scale and len(all_feature_matrices) > 1:
                max_clients = max(max(all_n_clients), max(cluster_sizes))
                ax_stats.set_ylim(0, max_clients + 1)
            
            # Add value labels on bars
            for bar, size in zip(bars, cluster_sizes):
                height = bar.get_height()
                ax_stats.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                             f'{size}', ha='center', va='bottom', fontweight='bold')
        
        # t-SNE Visualization
        if method in ['tsne', 'both']:
            if feature_matrix.shape[0] > 3:
                if fixed_scale and len(all_feature_matrices) > 1:
                    # Extract coordinates for this instance from global t-SNE
                    start_idx = sum(all_n_clients[:inst_idx])
                    end_idx = start_idx + n_clients
                    tsne_coords = global_tsne_coords[start_idx:end_idx]
                else:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, n_clients-1))
                    tsne_coords = tsne.fit_transform(feature_matrix)
                
                ax = axes[tsne_row, 0]
                
                for i, cluster_id in enumerate(unique_clusters):
                    mask = cluster_labels == cluster_id
                    ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                              c=[colors[i]], label=f'Cluster {cluster_id}', 
                              s=100, alpha=0.7, edgecolors='black', linewidth=1)
                    
                    # Add client ID annotations
                    for j, client_id in enumerate(client_ids[mask]):
                        ax.annotate(f'C{client_id}', 
                                   (tsne_coords[mask][j, 0], tsne_coords[mask][j, 1]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                # Set consistent scales if requested
                if fixed_scale and len(all_feature_matrices) > 1:
                    ax.set_xlim(tsne_x_range[0] - 0.1 * (tsne_x_range[1] - tsne_x_range[0]), 
                               tsne_x_range[1] + 0.1 * (tsne_x_range[1] - tsne_x_range[0]))
                    ax.set_ylim(tsne_y_range[0] - 0.1 * (tsne_y_range[1] - tsne_y_range[0]), 
                               tsne_y_range[1] + 0.1 * (tsne_y_range[1] - tsne_y_range[0]))
                
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_title(f't-SNE: {instance_names[inst_idx]} Clustering')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax = axes[tsne_row, 0]
                ax.text(0.5, 0.5, 'Too few clients\nfor t-SNE', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f't-SNE: {instance_names[inst_idx]} - Not enough data')
            
            if method == 'both':
                # Distance matrix visualization
                distance_matrix = squareform(pdist(feature_matrix, metric='euclidean'))
                
                ax_dist = axes[tsne_row, 1]
                
                # Use consistent color scale for distance matrices
                if fixed_scale and len(all_feature_matrices) > 1:
                    all_distances = [squareform(pdist(fm, metric='euclidean')) for fm in all_feature_matrices]
                    vmin_dist = min([dm.min() for dm in all_distances])
                    vmax_dist = max([dm.max() for dm in all_distances])
                else:
                    vmin_dist, vmax_dist = None, None
                
                im = ax_dist.imshow(distance_matrix, cmap='viridis', aspect='auto',
                                   vmin=vmin_dist, vmax=vmax_dist)
                ax_dist.set_xlabel('Client ID')
                ax_dist.set_ylabel('Client ID')
                ax_dist.set_title(f'{instance_names[inst_idx]} Distance Matrix')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax_dist)
                cbar.set_label('Euclidean Distance')
                
                # Add client labels
                ax_dist.set_xticks(range(n_clients))
                ax_dist.set_yticks(range(n_clients))
                ax_dist.set_xticklabels([f'C{i}' for i in range(n_clients)])
                ax_dist.set_yticklabels([f'C{i}' for i in range(n_clients)])
                
                # Cluster center analysis
                ax_centers = axes[tsne_row, 2]
                
                if hasattr(instance, 'cluster_centers_'):
                    centers = instance.cluster_centers_
                    
                    # Use consistent color scale for cluster centers
                    if fixed_scale and len(all_feature_matrices) > 1:
                        all_centers = [inst.cluster_centers_ for inst in fedcluster_instances 
                                     if hasattr(inst, 'cluster_centers_')]
                        if all_centers:
                            combined_centers = np.vstack(all_centers)
                            vmin_centers = combined_centers.min()
                            vmax_centers = combined_centers.max()
                        else:
                            vmin_centers, vmax_centers = None, None
                    else:
                        vmin_centers, vmax_centers = None, None
                    
                    sns.heatmap(centers, annot=True, cmap='Reds', fmt='.3f', 
                               ax=ax_centers, cbar_kws={'label': 'Center Value'},
                               vmin=vmin_centers, vmax=vmax_centers)
                    ax_centers.set_title(f'{instance_names[inst_idx]} Cluster Centers')
                    ax_centers.set_xlabel('Class')
                    ax_centers.set_ylabel('Cluster ID')
                else:
                    ax_centers.text(0.5, 0.5, 'Cluster centers\nnot available', 
                                   ha='center', va='center', transform=ax_centers.transAxes)
                    ax_centers.set_title(f'{instance_names[inst_idx]} Centers')
    
    plt.tight_layout()
    return fig

# Convenience function for comparing IID vs Non-IID
def compare_iid_noniid(iid_instance, noniid_instance, method='both', figsize=(20, 12)):
    """
    Convenience function to compare IID vs Non-IID clustering results
    
    Parameters:
    - iid_instance: FedCluster instance for IID data
    - noniid_instance: FedCluster instance for Non-IID data
    - method: 'pca', 'tsne', or 'both'
    - figsize: Figure size
    """
    return visualize_clustering([iid_instance, noniid_instance], 
                               method=method, 
                               figsize=figsize,
                               fixed_scale=True,
                               instance_names=['IID', 'Non-IID'])