import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.manifold import MDS

import skfuzzy as fuzz


class AutoAgglomerativeClustering:
    """
    Agglomerative Clustering with automatic cluster number detection
    """
    
    def __init__(self, method='silhouette', linkage='ward', 
                 min_clusters=2, max_clusters=10, metric='euclidean', fuzzy_clusters=3, fuzzy_thr=0.8, MDS_DIM=20):
        """
        Parameters:
        -----------
        method : str
            Method for automatic cluster detection:
            - 'silhouette': Maximize silhouette score
            - 'elbow': Elbow method on distances
            - 'gap': Gap statistic
            - 'dendrogram': Automatic cut based on distance threshold
        linkage : str
            Linkage criterion: 'ward', 'complete', 'average', 'single'
        min_clusters : int
            Minimum number of clusters to try
        max_clusters : int
            Maximum number of clusters to try
        metric : str
            Distance metric (only for non-ward linkage)
        """
        self.method = method
        self.linkage = linkage
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.metric = metric
        self.optimal_clusters_ = None
        self.labels_ = None
        self.linkage_matrix_ = None
        self.fuzzy_clusters = fuzzy_clusters
        self.fuzzy_thr = fuzzy_thr
        self.MDS_DIM = MDS_DIM
        self.scores_ = {}
        
    def fit(self, X):
        """Fit the clustering model and automatically detect optimal clusters"""
        
        if self.method == 'silhouette':
            self._fit_silhouette(X)
        elif self.method == 'elbow':
            self._fit_elbow(X)
        elif self.method == 'gap':
            self._fit_gap_statistic(X)
        elif self.method == 'dendrogram':
            self._fit_dendrogram(X)
        elif self.method == 'fuzzy_cmeans':
            self._fuzzy_Cmeans(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self

    def _fuzzy_Cmeans(self, dist_matrix):
        # X.shape : (K, K) k is # of clients 
        
        dist_matrix[dist_matrix < 0] = 0

        mds = MDS(n_components=self.MDS_DIM,
          dissimilarity='precomputed', # Crucial: tells MDS you're providing a distance matrix
          random_state=42,
          normalized_stress='auto')

        client_features = mds.fit_transform(dist_matrix)
        
        X = client_features.T
        cntr, Y, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X,
            c=self.fuzzy_clusters,
            m=2,
            error=0.005,
            maxiter=1000,
            init=None,
            seed=42
        )

        self.labels_ = [[] for _ in range(Y.shape[1])]
        for client_idx in range(Y.shape[1]):
            memberships = Y[:, client_idx]
            above_threshold_clusters = np.where(memberships >= self.fuzzy_thr)[0]

            if len(above_threshold_clusters) > 0:
                for cluster_idx in above_threshold_clusters:
                    self.labels_[client_idx].append(cluster_idx)
            else:
                best_cluster_idx = np.argmax(memberships)
                self.labels_[client_idx].append(best_cluster_idx)

    def _fit_silhouette(self, X):
        """Find optimal clusters using silhouette score"""
        silhouette_scores = []
        
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.linkage,
                metric=self.metric if self.linkage != 'ward' else 'euclidean'
            )
            labels = clusterer.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            
            self.scores_[n_clusters] = {
                'silhouette': score,
                'davies_bouldin': davies_bouldin_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels)
            }
        
        self.optimal_clusters_ = self.min_clusters + np.argmax(silhouette_scores)
        
        # Fit final model
        final_clusterer = AgglomerativeClustering(
            n_clusters=self.optimal_clusters_,
            linkage=self.linkage,
            metric=self.metric if self.linkage != 'ward' else 'euclidean'
        )
        self.labels_ = final_clusterer.fit_predict(X)
        
    def _fit_elbow(self, X):
        """Find optimal clusters using elbow method on within-cluster distances"""
        
        # Compute linkage matrix for dendrogram
        if self.linkage == 'ward':
            self.linkage_matrix_ = linkage(X, method='ward')
        else:
            distances = pdist(X, metric=self.metric)
            self.linkage_matrix_ = linkage(distances, method=self.linkage)
        
        inertias = []
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            labels = fcluster(self.linkage_matrix_, n_clusters, criterion='maxclust')
            
            # Calculate within-cluster sum of squares
            inertia = 0
            for cluster_id in np.unique(labels):
                cluster_points = X[labels == cluster_id]
                centroid = cluster_points.mean(axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)
            inertias.append(inertia)
        
        # Find elbow using second derivative
        inertias = np.array(inertias)
        if len(inertias) > 2:
            second_derivative = np.diff(inertias, 2)
            self.optimal_clusters_ = self.min_clusters + np.argmax(second_derivative) + 1
        else:
            self.optimal_clusters_ = self.min_clusters
        
        self.labels_ = fcluster(self.linkage_matrix_, self.optimal_clusters_, 
                               criterion='maxclust')
        
    def _fit_gap_statistic(self, X, n_refs=10):
        """Find optimal clusters using gap statistic"""
        
        gaps = []
        
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            # Compute observed within-cluster dispersion
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.linkage,
                metric=self.metric if self.linkage != 'ward' else 'euclidean'
            )
            labels = clusterer.fit_predict(X)
            
            wk = 0
            for cluster_id in np.unique(labels):
                cluster_points = X[labels == cluster_id]
                if len(cluster_points) > 1:
                    pairwise_dists = pdist(cluster_points, metric=self.metric)
                    wk += np.sum(pairwise_dists ** 2) / (2 * len(cluster_points))
            
            # Compute expected within-cluster dispersion (reference datasets)
            ref_wks = []
            for _ in range(n_refs):
                # Generate uniform reference data
                X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
                ref_clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=self.linkage,
                    metric=self.metric if self.linkage != 'ward' else 'euclidean'
                )
                ref_labels = ref_clusterer.fit_predict(X_ref)
                
                ref_wk = 0
                for cluster_id in np.unique(ref_labels):
                    cluster_points = X_ref[ref_labels == cluster_id]
                    if len(cluster_points) > 1:
                        pairwise_dists = pdist(cluster_points, metric=self.metric)
                        ref_wk += np.sum(pairwise_dists ** 2) / (2 * len(cluster_points))
                ref_wks.append(ref_wk)
            
            gap = np.log(np.mean(ref_wks)) - np.log(wk) if wk > 0 else 0
            gaps.append(gap)
        
        # Find first k where gap(k) >= gap(k+1) - std(k+1)
        gaps = np.array(gaps)
        self.optimal_clusters_ = self.min_clusters + np.argmax(gaps)
        
        # Fit final model
        final_clusterer = AgglomerativeClustering(
            n_clusters=self.optimal_clusters_,
            linkage=self.linkage,
            metric=self.metric if self.linkage != 'ward' else 'euclidean'
        )
        self.labels_ = final_clusterer.fit_predict(X)
        
    def _fit_dendrogram(self, X, inconsistency_threshold=0.7):
        """Find optimal clusters using dendrogram automatic cut"""
        
        # Compute linkage matrix
        if self.linkage == 'ward':
            self.linkage_matrix_ = linkage(X, method='ward')
        else:
            distances = pdist(X, metric=self.metric)
            self.linkage_matrix_ = linkage(distances, method=self.linkage)
        
        # Calculate distance threshold using inconsistency method
        # Cut at height where distance jump is largest
        distances = self.linkage_matrix_[:, 2]
        distance_diffs = np.diff(distances)
        
        # Find the largest gap in distances
        if len(distance_diffs) > 0:
            max_diff_idx = np.argmax(distance_diffs)
            threshold = distances[max_diff_idx] + distance_diffs[max_diff_idx] * inconsistency_threshold
            
            # Get clusters
            self.labels_ = fcluster(self.linkage_matrix_, threshold, criterion='distance')
            self.optimal_clusters_ = len(np.unique(self.labels_))
        else:
            self.optimal_clusters_ = 1
            self.labels_ = np.zeros(len(X), dtype=int)
    
    def fit_predict(self, X):
        """Fit the model and return cluster labels"""
        self.fit(X)
        return self.labels_
    
    def plot_dendrogram(self, figsize=(12, 6)):
        """Plot dendrogram if available"""
        if self.linkage_matrix_ is None:
            print("Linkage matrix not computed. Use method='dendrogram' or 'elbow'")
            return
        
        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix_)
        plt.title(f'Hierarchical Clustering Dendrogram (Optimal clusters: {self.optimal_clusters_})')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    
    def plot_evaluation_metrics(self, figsize=(15, 5)):
        """Plot evaluation metrics across different cluster numbers"""
        if not self.scores_:
            print("No scores computed. Use method='silhouette' first.")
            return
        
        n_clusters_range = sorted(self.scores_.keys())
        
        metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for idx, metric in enumerate(metrics):
            scores = [self.scores_[n][metric] for n in n_clusters_range]
            axes[idx].plot(n_clusters_range, scores, 'bo-')
            axes[idx].axvline(self.optimal_clusters_, color='r', linestyle='--', 
                            label=f'Optimal: {self.optimal_clusters_}')
            axes[idx].set_xlabel('Number of Clusters')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Score')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage and comparison
if __name__ == "__main__":
    # Generate sample data with different cluster characteristics
    np.random.seed(42)
    
    # Create data with varying cluster sizes (simulating Non-IID FL scenario)
    X1, y1 = make_blobs(n_samples=100, centers=3, n_features=2, 
                        cluster_std=0.5, random_state=42)
    X2, y2 = make_blobs(n_samples=30, centers=1, n_features=2, 
                        cluster_std=0.3, random_state=43)
    X2 = X2 + np.array([8, 8])  # Move outlier cluster
    
    # Add some outliers
    outliers = np.random.uniform(-2, 12, (10, 2))
    
    X = np.vstack([X1, X2, outliers])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compare different methods
    methods = ['silhouette', 'elbow', 'gap', 'dendrogram']
    results = {}
    
    print("Automatic Cluster Detection Results:")
    print("=" * 50)
    
    for method in methods:
        clusterer = AutoAgglomerativeClustering(
            method=method,
            linkage='ward',
            min_clusters=2,
            max_clusters=8
        )
        labels = clusterer.fit_predict(X_scaled)
        
        results[method] = {
            'n_clusters': clusterer.optimal_clusters_,
            'labels': labels,
            'clusterer': clusterer
        }
        
        print(f"\n{method.upper()} Method:")
        print(f"  Optimal clusters: {clusterer.optimal_clusters_}")
        print(f"  Cluster sizes: {np.bincount(labels)}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        labels = results[method]['labels']
        n_clusters = results[method]['n_clusters']
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_title(f'{method.upper()}: {n_clusters} clusters', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show detailed metrics for silhouette method
    print("\n" + "=" * 50)
    print("Detailed Evaluation Metrics (Silhouette Method):")
    print("=" * 50)
    
    silhouette_clusterer = results['silhouette']['clusterer']
    silhouette_clusterer.plot_evaluation_metrics()
    
    # Show dendrogram
    if results['dendrogram']['clusterer'].linkage_matrix_ is not None:
        results['dendrogram']['clusterer'].plot_dendrogram()