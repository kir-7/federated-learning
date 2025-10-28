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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from autoAgglo import AutoAgglomerativeClustering

from dataclasses import dataclass
from tqdm.auto import tqdm
from copy import deepcopy
from models import MNISTModel, CIFAR10Model, ResnetModel

'''
This will be used for clustering based on model state, this approach requires to maintain a seperate client model all the time.  
'''

@dataclass
class FedStateClusterConfig:
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
    
    # Agglomarative clustering parameters
    aggl_method : str = 'silhouette'
    aggl_linkage : str = 'ward' 

    cluster_every : int  = 3
    start_recluster : int = 20
    local_eval_every : int  = 3

    verbose: bool = True


class FedStateCluster:
    def __init__(self, config: FedStateClusterConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(self.config.torch_seed)        
        np.random.seed(self.config.np_seed)
        
        self.n_clients = config.n_clients
        self.global_rounds = config.global_rounds
        
        if config.dataset == 'cifar10':
            self.train_transform, self.val_transform = self.get_transforms_cifar10()
        elif config.dataset == "mnist":
            self.train_transform, self.val_transform = self.get_transforms_mnist()
        elif config.dataset == 'rotmnist':
            self.train_transform, self.simple_train_transform, self.val_transform = self.get_transforms_rotmnist()
        else:
            raise ValueError(f"dataset: {config.dataset} not allowed")
    
        self.logger = {} 
        
        self.setup_dataset()
        
        # first create models for all clients
        self.setup_models()

        # then assign those models to clusters
        self.initialize_clusters()

    def get_transforms_rotmnist(self):

        simple_train_transform = transforms.Compose([
            transforms.ToTensor(),         
            transforms.Normalize(0.1307, 0.3081) # Normalizes the tensor
        ])

        train_transform = transforms.Compose([
            transforms.RandomRotation((180, 180)), 
            transforms.ToTensor(),         
            transforms.Normalize(0.1307, 0.3081) # Normalizes the tensor
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),         
            transforms.Normalize(0.1307, 0.3081) # Normalizes the tensor
        ])

        return train_transform, simple_train_transform, val_transform

    
    def get_transforms_mnist(self):
        
        train_transform = transforms.Compose([
            # transforms.RandomRotation(10), 
            transforms.ToTensor(),         
            transforms.Normalize(0.1307, 0.3081) # Normalizes the tensor
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),         
            transforms.Normalize(0.1307, 0.3081) # Normalizes the tensor
        ])

        return train_transform, val_transform


    def get_transforms_cifar10(self):
        
        # testing out Rotated MNist performance
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

    def apply_transforms_simple_train(self, batch):
        if self.config.dataset == 'flwrlabs/femnist':
            batch['img'] = [self.simple_train_transform(img.convert("L")) for img in batch['image']] # Convert to grayscale
            batch['label'] = batch['character']
            del batch['image']
            del batch['character']
        if self.config.dataset == 'mnist' or self.config.dataset == 'rotmnist':
            batch['img'] = [self.simple_train_transform(img) for img in batch['image']]
            del batch['image']
        else:
            batch['img'] = [self.simple_train_transform(img) for img in batch['img']]
        return batch
    
    def apply_transforms_train(self, batch):
        if self.config.dataset == 'flwrlabs/femnist':
            batch['img'] = [self.train_transform(img.convert("L")) for img in batch['image']] # Convert to grayscale
            batch['label'] = batch['character']
            del batch['image']
            del batch['character']
        if self.config.dataset == 'mnist' or self.config.dataset == 'rotmnist':
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
        if self.config.dataset == 'mnist' or self.config.dataset == 'rotmnist':
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
                dataset='mnist' if self.config.dataset == 'rotmnist' else self.config.dataset,
                partitioners={
                    "train": self.partitioner,
                    'test': deepcopy(self.partitioner)
                },
                trust_remote_code=True
            )
            
            self.client_partitions_test = {}
            self.client_partitions_train = {}

            for i in range(self.n_clients):
            
                if i < int(0.5 * self.n_clients):
                    self.client_partitions_train[i] =  fds.load_partition(i, "train").with_transform(self.apply_transforms_simple_train)                    
                    self.client_partitions_test[i] =  fds.load_partition(i, "test").with_transform(self.apply_transforms_test)  # dont apply rotation to test set                
                else:
                    self.client_partitions_train[i] =  fds.load_partition(i, "train").with_transform(self.apply_transforms_train)
                    self.client_partitions_test[i] =  fds.load_partition(i, "test").with_transform(self.apply_transforms_train)  # apply rotation to test st as well
                
                   
            # Load the global test partition for other datasets
            self.global_test_partition = fds.load_split('test').with_transform(self.apply_transforms_test)
        
        # Make this more flexible for different datasets
        self.all_labels = sorted(range(0, self.config.n_classes))  # TODO: make this dataset-dependent   

        self.client_loaders_train = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=True) for i, cp in self.client_partitions_train.items() }
        self.client_loaders_test = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=False)  for i, cp in self.client_partitions_test.items() }
        
        self.global_test_loader = DataLoader(self.global_test_partition, batch_size=self.config.global_bs)
    
    def setup_models(self):
        # Create models for each cluster and move to device
        if self.config.dataset == 'mnist' or self.config.dataset == 'rotmnist' or self.config.dataset == 'flwrlabs/femnist':
            self.models = [MNISTModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]
        else:
            self.models = [ResnetModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]        
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def global_scheduler(self, round_num):
        # simple: every 2 global rounds multiple the lr by a factor of 0.9  
        return self.config.client_lr * (0.9**(round_num//2))

    def create_agglomarative_clusters(self, similarities):
        '''
        Create Clustering using Agglomarative clustering
        '''    
        
        distance_matrix = 1 - similarities
        
        aggl_clusterer = AutoAgglomerativeClustering(
            method='silhouette',
            metric='precomputed',
            linkage='complete', # mostly will be `complete`
            min_clusters=2,  # if clustering based on model state then before starting mostly all models should be in same cluster, then they should start to diverge  
            max_clusters = min(10, self.config.n_clients-1)
        )
        
        self.labels_ = aggl_clusterer.fit_predict(distance_matrix)
        self.aggl_model = aggl_clusterer
        self.config.n_clusters = max(self.labels_)+1

        return self.labels_

    @staticmethod
    def flatten(source : dict):
        return torch.cat([value.flatten() for value in source.values()])

    @staticmethod
    def pairwise_similarity(vectors):        
        # get angles between all vector pairs
        norm_vectors = F.normalize(torch.stack(vectors), p=2, dim=1)
        angles = norm_vectors @ norm_vectors.T
        return angles.cpu().numpy()
    
    def initialize_clusters(self):        
        print("Initializing with a single cluster for all clients.")
        self.config.n_clusters = 1
        self.clusters = {0:list(range(self.config.n_clients))}

    def assign_cluster_model_conv(self):
        '''
        assign each of n clients to p clusters
        APPROACH: assign based on the model's state, this will be called every few global rounds to recluster the clients   
        '''
        
        model_states = [model.state_dict() for model in self.models]

        flattend_states = [self.flatten(model_state) for model_state in model_states]

        similarities = self.pairwise_similarity(flattend_states)
                
        assigned_clusters = self.create_agglomarative_clusters(similarities)

        # assign new clusters
        self.clusters = {i: [] for i in range(self.config.n_clusters)}
        for client in range(self.n_clients):
            self.clusters[assigned_clusters[client]].append(client)        
        
        if self.config.verbose:
            print(f"Finished clustering clients.\nCluster Assignment: {self.clusters}")            

    @staticmethod
    def cluster_fed_avg(local_models):
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
            # NOTE: Tensors like 'num_batches_tracked' in BatchNorm should not be averaged.
            # They are integers. We can just keep the one from the last model.
            # A simple check for floating point type ensures we only average weights, biases, and running stats.
            if avg_state_dict[key].dtype == torch.float32 or avg_state_dict[key].dtype == torch.float64:
                avg_state_dict[key] = torch.div(avg_state_dict[key], len(local_models))
        
        # return th aggregated state dict
        return avg_state_dict 

    @torch.inference_mode()
    def evaluate_global(self, dataloader):
        # Get one representative model from each cluster (they are identical post-aggregation)
        cluster_models = []

        for cl in range(self.config.n_clusters):
            if self.clusters[cl]: # Ensure cluster is not empty
                rep_client_id = self.clusters[cl][0]
                cluster_models.append(self.models[rep_client_id])
        
        if not cluster_models:
             print("No models to evaluate.")
             return {'global_accuracy': 0, 'cluster_win_rates': []}
        
        for model in cluster_models:
            model.eval()
        
        total_samples = 0
        correct_predictions = 0
        all_best_clusters = []
        
        for sample in tqdm(dataloader, desc="Global Evaluation", leave=False):
            images, labels = sample['img'], sample['label']
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            
            best_preds_per_cluster = torch.zeros(batch_size, len(cluster_models), dtype=torch.long, device=self.device)
            best_probs_per_cluster = torch.zeros(batch_size, len(cluster_models), device=self.device)
            
            for i, model in enumerate(cluster_models):
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, max_indices = torch.max(probs, dim=1)
                best_preds_per_cluster[:, i] = max_indices
                best_probs_per_cluster[:, i] = max_probs
                
            winning_cluster_indices = torch.argmax(best_probs_per_cluster, dim=1)
            final_predictions = best_preds_per_cluster.gather(1, winning_cluster_indices.unsqueeze(-1)).squeeze(-1)
            
            correct_predictions += (final_predictions == labels).sum().item()
            total_samples += batch_size
            all_best_clusters.extend(winning_cluster_indices.cpu().numpy())

        global_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        cluster_wins = np.bincount(all_best_clusters, minlength=len(cluster_models))
        cluster_win_rates = cluster_wins / total_samples if total_samples > 0 else np.zeros(len(cluster_models))
        
        if self.config.verbose:
            print(f"Global Test Accuracy: {global_accuracy:.4f}")
            print(f"Cluster Win Rates: {[f'{rate:.2%}' for rate in cluster_win_rates]}")

        return {
            'global_accuracy': global_accuracy,
            'cluster_win_rates': cluster_win_rates.tolist(),
        }

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
    
    def train(self, client, model, lr, dataloader):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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
        
        return {
            'avg_train_loss': sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0,
            "train_loss": epoch_loss[-1] if epoch_loss else 0
        }
    
    def run(self):

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
                                
                # Train selected clients
                for client in selected_clients:
                    lr = self.global_scheduler(round_num)
                    train_res = self.train(
                        client, self.models[client], lr, self.client_loaders_train[client]
                    )
                    
                    result[f'client_{client}'] = train_res
                
                # aggregate
                if round_num % self.config.cluster_every != 0 or round_num < self.config.start_recluster:
                    aggregated_model_state = self.cluster_fed_avg([self.models[client] for client in selected_clients])
                    for client in self.clusters[cl]: self.models[client].load_state_dict(aggregated_model_state)                                                    
                                
            # after all clusters are trained then re cluster this round if allowed            
            if round_num % self.config.cluster_every == 0 and round_num >= self.config.start_recluster:
                print(f"Round: {round_num} --- Reclustering")
                self.assign_cluster_model_conv()
    
                for cl in range(self.config.n_clusters):
                    aggregated_model_state = self.cluster_fed_avg([self.models[client] for client in self.clusters[cl]])    
                    for client in self.clusters[cl]: self.models[client].load_state_dict(aggregated_model_state)
                                                    
            # Evaluate clients on their test sets if test set exists (independed test set does not exist for femnist)
            # evaluate each client only once every 3 global itertions
            if round_num % self.config.local_eval_every == 0 or round_num == self.config.global_rounds-1:
                cluster_evals = []
    
                for cl in range(self.config.n_clusters):
                    client_evals = []
                    for client in self.clusters[cl]:
                        if client in self.client_loaders_test: 
                            eval_res = self.evaluate(client, cl, self.models[client], self.client_loaders_test[client])
                            result[f'client_{client}'].update(**eval_res)
                            client_evals.append(eval_res['val_acc'])

                    cluster_evals.append(sum(client_evals)/len(client_evals))
                
                # report the average client performance across all clusters this might be more important than global test loader performance
                print(f"Average all clients performance: {sum(cluster_evals)/len(cluster_evals)}")            
                result.update({"average_acc":sum(cluster_evals)/len(cluster_evals)})
            
            # for each data sample in the global test set we obtain the highest probable class from each cluster and then pick the highest probable among those and verify it with ground truth
            # This global evaluation is not really that important in Model state based clusteirng
            eval_global_res = self.evaluate_global(self.global_test_loader)
            result.update(**eval_global_res)

            if self.config.verbose:
                print(result)
            
            history.append(result)
        
            self.logger[round_num] = {"cluster":self.clusters, **result}            

        return history, self.clusters
