import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from torchvision import transforms
from datasets import load_dataset

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.metrics.utils import compute_counts
from flwr_datasets.visualization import plot_label_distributions

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, DBSCAN  # Changed from AgglomerativeClustering

from dataclasses import dataclass
from tqdm.auto import tqdm
from copy import deepcopy
from models import MNISTModel, CIFAR10Model, ResnetModel

@dataclass
class FedClusterConfig:
    n_clients: int = 10
    n_clusters: int = 2
    m: float = 1.0    # fraction of clients of each cluster that needs to be involved in training
    
    method: str = 'dirichlet'
    dirichlet_alpha: float = 0.3
    dataset: str = "cifar10"    
    partition_column : str = 'label'
    n_classes : int = 10

    client_bs: int = 32
    global_bs: int = 32
    
    global_rounds: int = 10
    client_epochs: int = 5
    client_lr: float = 1e-4
    step_size : int = 2
    lr_schd_gamma = 0.5
    
    # seed
    torch_seed: int = 42
    np_seed: int = 43
    
    # K-Means specific parameters
    kmeans_init: str = 'k-means++'  # initialization method for K-Means
    kmeans_max_iter: int = 300      # maximum iterations for K-Means
    kmeans_tol: float = 1e-4        # tolerance for K-Means convergence
    
    # DBSCAN clustering parameters
    dbscan_eps : float = 0.5
    dbscan_min_samples : int = 3

    # Agglomarative clustering parameters
    use_agglomarative : bool = False
    aggl_method : str = 'silhouette'
    aggl_linkage : str = 'ward' 

    # DBSCAN Parameters
    dbscan_eps : float = 0.5
    dbscan_min_samples : int = 2
    use_dbscan : bool = False

    verbose: bool = True

class FedCluster:
    def __init__(self, config: FedClusterConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(self.config.torch_seed)        
        np.random.seed(self.config.np_seed)
        
        self.n_clients = config.n_clients
        self.global_rounds = config.global_rounds
        
        self.train_transform, self.val_transform = self.get_transforms()
        
        self.setup_dataset()
        self.assign_clusters_class_distribution()
        self.setup_models()
        
    def get_transforms(self):
        
        train_transform = transforms.Compose([
            transforms.ToTensor(),            
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) if self.config.dataset == 'cifar10' else transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  if self.config.dataset == 'cifar10' else transforms.Normalize((0.1307,), (0.3081,))
        ])

        return train_transform, val_transform

    
    def apply_transforms_train(self, batch):
        if self.config.dataset == 'flwrlabs/femnist':
            batch['img'] = [self.train_transform(img.convert("L")) for img in batch['image']] # Convert to grayscale
            batch['label'] = batch['character']
            del batch['image']
            del batch['character']
        if self.config.dataset == 'mnist':
            batch['img'] = [self.train_transform(img) for img in batch['image']]
            del batch['image']
        else:
            batch['img'] = [self.train_transform(img) for img in batch['img']]
        return batch
    
    def apply_transforms_test(self, batch):
        if self.config.dataset == 'flwrlabs/femnist':
            batch['img'] = [self.val_transform(img.convert("L")) for img in batch['image']] # Convert to grayscale
            batch['label'] = batch['character']
            del batch['image']
            del batch['character']
        if self.config.dataset == 'mnist':
            batch['img'] = [self.val_transform(img) for img in batch['image']]
            del batch['image']
        else:
            batch['img'] = [self.val_transform(img) for img in batch['img']]
        return batch

    def setup_dataset(self):
        if self.config.method == 'dirichlet':
            self.partitioner = DirichletPartitioner(
                num_partitions=self.config.n_clients,
                partition_by=self.config.partition_column,
                alpha=self.config.dirichlet_alpha,
                min_partition_size=10
            )
        elif self.config.method == 'iid':
            self.partitioner = IidPartitioner(num_partitions=self.config.n_clients)
        else:
            raise ValueError(f"Unknown partitioning method: {self.config.method}")

        if self.config.dataset == 'flwrlabs/femnist':

            fds = FederatedDataset(
                dataset=self.config.dataset,
                partitioners={"train": self.partitioner},
                trust_remote_code=True
            )
            
            # for femnist create train/test split from the train data for each client
            self.client_partitions_train = {}
            self.client_partitions_test = {}
            
            for i in range(self.config.n_clients):
                full_partition = fds.load_partition(i, "train")
                
                # Split each client's data into train/test (80/20)
                total_size = len(full_partition)
                train_size = int(0.8 * total_size)
                
                train_indices = list(range(train_size))
                test_indices = list(range(train_size, total_size))
                
                self.client_partitions_train[i] = full_partition.select(train_indices).with_transform(self.apply_transforms_train)
                self.client_partitions_test[i] = full_partition.select(test_indices).with_transform(self.apply_transforms_test)

            all_test_data = []
            for i in range(self.config.n_clients):
                client_test = fds.load_partition(i, "train")
                total_size = len(client_test)
                train_size = int(0.8 * total_size)
                test_indices = list(range(train_size, total_size))
                all_test_data.extend([client_test[idx] for idx in test_indices])
            
            # Convert to dataset format for global test
            from datasets import Dataset
            self.global_test_partition = Dataset.from_list(all_test_data).with_transform(self.apply_transforms_test)
            
        else:
            fds = FederatedDataset(
                dataset=self.config.dataset,
                partitioners={
                    "train": self.partitioner,
                    'test': deepcopy(self.partitioner)
                },
                trust_remote_code=True
            )
        
            self.client_partitions_train = {
                i: fds.load_partition(i, "train").with_transform(self.apply_transforms_train) 
                for i in range(self.n_clients)
            }
            self.client_partitions_test = {
                i: fds.load_partition(i, 'test').with_transform(self.apply_transforms_test)
                for i in range(self.n_clients)
            }
       
            # Load the global test partition for other datasets
            self.global_test_partition = fds.load_split('test').with_transform(self.apply_transforms_test)
        
        # Make this more flexible for different datasets
        self.all_labels = sorted(range(0, self.config.n_classes))  # TODO: make this dataset-dependent   

        self.client_loaders_train = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=True) for i, cp in self.client_partitions_train.items() }
        self.client_loaders_test = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=False)  for i, cp in self.client_partitions_test.items() }
        
        self.global_test_loader = DataLoader(self.global_test_partition, batch_size=self.config.global_bs)
    
    def setup_models(self):
        # Create models for each cluster and move to device
        if self.config.dataset == 'mnist' or self.config.dataset == 'flwrlabs/femnist':
            self.models = [MNISTModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clusters)]
        else:
            self.models = [CIFAR10Model(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clusters)]        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def get_distribution_stats(self):
        df = compute_counts(self.partitioner, 'label')
       
        dist_stats = []
        for c in range(self.n_clients):
            partition_size = len(self.client_partitions_train[c])
            if partition_size == 0:   
                client_dist = {label: 0.0 for label in range(df.shape[1])}
            else:  
                client_dist = {label: df.iloc[c][label] / partition_size for label in range(df.shape[1])}
            dist_stats.append(client_dist)
        
        return dist_stats
    
    def create_feature_matrix(self, distribution_stats):
        """
        Create a feature matrix from distribution statistics for K-Means clustering.
        Each row represents a client, each column represents a class probability.
        """
        n_classes = len(self.all_labels)
        feature_matrix = np.zeros((self.n_clients, n_classes))
        
        for i, dist in enumerate(distribution_stats):
            for j, label in enumerate(self.all_labels):
                feature_matrix[i, j] = dist.get(label, 0.0)
        
        return feature_matrix
    
    def create_clusters(self, distribution_stats):
        """
        Use K-Means clustering on the class distribution features.
        """
        # Create feature matrix where each client is represented by their class distribution
        feature_matrix = self.create_feature_matrix(distribution_stats)
        
        # Apply K-Means clustering
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            init=self.config.kmeans_init,
            max_iter=self.config.kmeans_max_iter,
            tol=self.config.kmeans_tol,
            random_state=self.config.np_seed,
            n_init=10  # number of random initializations
        )
        
        self.labels_ = kmeans.fit_predict(feature_matrix)
        self.kmeans_model_ = kmeans  # Store the fitted model
        self.cluster_centers_ = kmeans.cluster_centers_  # Store cluster centers
        self.inertia_ = kmeans.inertia_  # Store within-cluster sum of squares

        if self.config.verbose:
            print(f"K-Means inertia (within-cluster sum of squares): {self.inertia_:.4f}")
            print(f"Cluster centers shape: {self.cluster_centers_.shape}")
      
        
        return self.labels_

    def create_dbscan_clustering(self, distribution_stats):
        '''
        Create Clusters using DBSCAN
        '''
        
        feature_matrix = self.create_feature_matrix(distribution_stats)

        dbs = DBSCAN(self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples)

        self.labels_ = dbs.fit_predict(feature_matrix)
        self.dbs_model = dbs

        count = max(self.labels_)+1

        # all clients classified as noise will be their own clusters
        for i in range(self.config.n_clients):
            if self.labels_[i] == -1:
                self.labels_[i] = count
                count += 1        
        
        self.config.n_clusters = count

        return self.labels_    

    def create_agglomarative_clusters(self, distribution_stats):
        '''
        Create Clustering using Agglomarative clustering
        '''
        from autoAgglo import AutoAgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        
        feature_matrix = self.create_feature_matrix(distribution_stats)
        scaler = StandardScaler()

        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        aggl_clusterer = AutoAgglomerativeClustering(
            method=self.config.aggl_method,
            linkage=self.config.aggl_linkage,
            min_clusters=min(10, self.config.n_clusters),
            max_clusters = min(10, self.config.n_clients-1)
        )
        
        self.labels_ = aggl_clusterer.fit_predict(feature_matrix_scaled)
        self.aggl_model = aggl_clusterer
        self.config.n_clusters = max(self.labels_)+1
        
        return self.labels_

    def assign_clusters_class_distribution(self):  
        '''
        assign each of n clients to p clusters
        APPROACH: assign based on data(class) distributions using K-Means
        '''
        
        distribution_stats = self.get_distribution_stats()
        if self.config.use_agglomarative:
            assigned_clusters = self.create_agglomarative_clusters(distribution_stats)
        elif self.config.use_dbscan:
                assigned_clusters = self.create_dbscan_clustering(distribution_stats)
        else:
            assigned_clusters = self.create_clusters(distribution_stats) 

        self.clusters = {i: [] for i in range(self.config.n_clusters)}
        
        for client in range(self.n_clients):
            self.clusters[assigned_clusters[client]].append(client)
        
        if self.config.verbose:
            print(f"Cluster Assignment: {self.clusters}")
            
            # Print cluster statistics
            for cluster_id in range(self.config.n_clusters):
                clients_in_cluster = self.clusters[cluster_id]
                print(f"Cluster {cluster_id}: {len(clients_in_cluster)} clients - {clients_in_cluster}")
                
                # Print average class distribution for this cluster
                if clients_in_cluster:
                    avg_dist = np.mean([
                        [distribution_stats[c].get(label, 0.0) for label in self.all_labels] 
                        for c in clients_in_cluster
                    ], axis=0)
                    print(f"  Average class distribution: {avg_dist}")
    
    @staticmethod
    def cluster_fed_avg(local_models, global_model):
        if not local_models:
            return
            
        # Get the state_dict of the first local model to initialize the average
        avg_state_dict = deepcopy(local_models[0].state_dict())
        
        # Sum the state_dicts of all other models
        for i in range(1, len(local_models)):
            local_state_dict = local_models[i].state_dict()
            for key in avg_state_dict:
                avg_state_dict[key] += local_state_dict[key]
                
        # Average the state_dict
        for key in avg_state_dict:
            # Note: Tensors like 'num_batches_tracked' in BatchNorm should not be averaged.
            # They are integers. We can just keep the one from the last model.
            # A simple check for floating point type ensures we only average weights, biases, and running stats.
            if avg_state_dict[key].dtype == torch.float32 or avg_state_dict[key].dtype == torch.float64:
                avg_state_dict[key] = torch.div(avg_state_dict[key], len(local_models))

        # Update the global model with the averaged state_dict
        global_model.load_state_dict(avg_state_dict)

    @torch.inference_mode()
    def evaluate_global(self, models, dataloader):
        """
        Evaluate all cluster models on global test set.
        For each sample, get probability distributions from all clusters,
        then predict the class with highest probability across all clusters.
        """
        
        for model in models:
            model.eval()
        
        all_predictions = []
        all_labels = []
        all_prob_matrices = []
        all_best_clusters = []
        
        total_samples = 0
        correct_predictions = 0
        
        # Store per-cluster predictions for analysis
        cluster_predictions = [[] for _ in range(self.config.n_clusters)]
        cluster_correct = [0 for _ in range(self.config.n_clusters)]
        
        for sample in tqdm(dataloader, desc=f"Global Evaluation", leave=False):
            images, labels = sample['img'], sample['label']
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            
            # Shape: [batch_size, n_clusters, n_classes]
            prob_matrix = torch.zeros(batch_size, self.config.n_clusters, self.config.n_classes, device=self.device)
            
            # Get predictions from each cluster model
            for i, model in enumerate(models):
                outputs = model(images)
                prob_matrix[:, i, :] = F.softmax(outputs, dim=1)
            
            # Single-step approach: find global maximum across all clusters and classes
            # Reshape to [batch_size, n_clusters * n_classes] and find global max
            flattened_probs = prob_matrix.view(batch_size, -1)
            global_max_indices = torch.argmax(flattened_probs, dim=1)
            
            # Extract final predictions and best clusters
            final_predictions = global_max_indices % self.config.n_classes
            best_clusters = global_max_indices // self.config.n_classes
            
            # Calculate accuracy
            correct_mask = (final_predictions == labels)
            correct_predictions += correct_mask.sum().item()
            total_samples += batch_size
            
            # Store results
            all_predictions.extend(final_predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_best_clusters.extend(best_clusters.cpu().numpy())
            all_prob_matrices.append(prob_matrix.cpu().numpy())
            
            # Per-cluster analysis (optional)
            for cluster_idx in range(self.config.n_clusters):
                cluster_preds = torch.argmax(prob_matrix[:, cluster_idx, :], dim=1)
                cluster_predictions[cluster_idx].extend(cluster_preds.cpu().numpy())
                cluster_correct[cluster_idx] += (cluster_preds == labels).sum().item()
        
        # Calculate final metrics
        global_accuracy = correct_predictions / total_samples
        
        # Per-cluster accuracies (for analysis)
        cluster_accuracies = [correct / total_samples for correct in cluster_correct]
        
        # Aggregate all probability matrices
        all_prob_matrices = np.vstack(all_prob_matrices)
        
        # Analyze cluster contribution
        cluster_wins = np.bincount(all_best_clusters, minlength=self.config.n_clusters)
        cluster_win_rates = cluster_wins / total_samples
        
        print(f"Global Test Accuracy: {global_accuracy:.4f}")
        print(f"Individual Cluster Accuracies: {[f'{acc:.4f}' for acc in cluster_accuracies]}")
        print(f"Cluster Win Rates: {[f'{rate:.2%}' for rate in cluster_win_rates]}")
        
        # Results dictionary
        results = {
            'global_accuracy': global_accuracy,
            'cluster_accuracies': cluster_accuracies,
            'cluster_win_rates': cluster_win_rates,            
        }
        
        return results



    @torch.inference_mode()
    def evaluate(self, client, clus, model, dataloader):
        model.eval()
        
        correct = 0
        total = 0
        test_loss = 0
        
        for sample in tqdm(dataloader, desc=f"Validating Cluster: {clus} clinet: {client}", leave=False):
            images, labels = sample['img'], sample['label']
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            test_loss += self.criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = test_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        return {"val_acc": accuracy, "val_avg_loss": avg_loss}
    
    def train(self, client, model, dataloader):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.client_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.lr_schd_gamma)
        
        epoch_loss = []
        
        for epoch in tqdm(range(self.config.client_epochs), desc=f"Training clinet: {client}", leave=False):
            batch_loss = []
            
            for batch_idx, sample in enumerate(dataloader):
                
                images, labels = sample['img'], sample['label']
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()  
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            
            scheduler.step()
            
            avg_loss = sum(batch_loss) / len(batch_loss) if batch_loss else 0
            epoch_loss.append(avg_loss)
        
        return model, {
            'avg_train_loss': sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0,
            "train_loss": epoch_loss[-1] if epoch_loss else 0
        }
    
    def run(self):
        '''
        this will follow the structure:
            - for each client: get the latest model assigned to their cluster; train on assigned dataset; use different optimizers; and store the updated weights
            - for each cluster : aggregated the cluster's local models and apply FedAVG on that. 
            - for each cluster : evaluate the cluster model on test dataset and take the least loss and get corresponding accuracy
            - for each cluster : evaluate the cluster on the specific cluster test dataset and report te accuracy for each cluster    
        '''

        history = []
        
        for round_num in range(self.config.global_rounds):
            
            if self.config.verbose:
                print(f"\n=== Global Round {round_num + 1}/{self.config.global_rounds} ===")
            
            result = {}
            
            for cl in range(self.config.n_clusters):
                if not self.clusters[cl]:  # Skip empty clusters
                    continue
                
                # Select clients for this round
                m = max(int(self.config.m * len(self.clusters[cl])), 1)
                selected_clients = np.random.choice(self.clusters[cl], m, replace=False).tolist()
                
                local_models = []
                
                # Train selected clients
                for client in selected_clients:
                    local_trained_model, train_res = self.train(
                        client,
                        deepcopy(self.models[cl]), 
                        self.client_loaders_train[client]
                    )

                    local_models.append(local_trained_model)
                    
                    result[f'client_{client}'] = train_res
                
                # Aggregate models
                self.cluster_fed_avg(local_models, self.models[cl])
                
                # Evaluate clients on their test sets if test set exists (independed test set does not exist for femnist)
                for client in selected_clients:
                    if client in self.client_loaders_test: 
                        eval_res = self.evaluate(client, cl, self.models[cl], self.client_loaders_test[client])
                        result[f'client_{client}'].update(**eval_res)
                

            # for each data sample in the global test set we obtain the highest probable class from each cluster and then pick the highest probable among those and verify it with ground truth
            eval_global_res = self.evaluate_global([self.models[i] for i in range(self.config.n_clusters)], self.global_test_loader)
            
            result.update(**eval_global_res)

            if self.config.verbose:
                print(result)
            
            history.append(result)
        
        return history, self.clusters

if __name__ == '__main__':
    config = FedClusterConfig()
    component = FedCluster(config)
    results, clusters = component.run()