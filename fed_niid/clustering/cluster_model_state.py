import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets import load_dataset

import numpy as np
from autoAgglo import AutoAgglomerativeClustering

from dataclasses import dataclass, asdict
from tqdm.auto import tqdm
from copy import deepcopy
from models import MNISTModel, ResnetModel, FemnistModel

import math

import copy
import json


'''
This will be used for clustering based on model state, this approach requires to maintain a seperate client model all the time.  
'''

@dataclass
class FedStateClusterConfig:
    n_clients: int = 10
    n_clusters: int = 1
    max_clusters : int = 2
    min_clusters : int = 3
    # fuzz_clusters : int = 1  # set it to 1 if not being used
    m: float = 1.0    # fraction of clients of each cluster that needs to be involved in training
    
    method: str = 'dirichlet'
    dirichlet_alpha: float = 0.3
    dataset: str = "cifar10"    
    n_classes : int = 10
    model : str = 'mnist'

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
    
    # Agglomarative clustering parameters
    aggl_method : str = 'silhouette'
    aggl_linkage : str = 'ward' 
    # fuzzy_thr : float=0.8
    

    cluster_every : int  = 3
    start_recluster : int = 20
    local_eval_every : int  = 3

    verbose: bool = True

class FedStateCluster:
    def __init__(self, config: FedStateClusterConfig, global_test_partition:dict[str, torch.utils.data.Dataset], client_parition_train:dict[str, torch.utils.data.Dataset], client_partitions_test:dict[str, torch.utils.data.Dataset]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(self.config.torch_seed)        
        np.random.seed(self.config.np_seed)
        
        self.n_clients = config.n_clients
        self.global_rounds = config.global_rounds
                
        self.logger = {} 
        self.similarity_logs = {}
        
        self.client_partitions_test = client_partitions_test
        self.client_partitions_train = client_parition_train
        self.global_test_partition = global_test_partition

        self.setup_dataset()
        
        # first create models for all clients
        self.setup_models()

        # then assign those models to clusters
        self.initialize_clusters()

    def setup_dataset(self):        
        
        # Make this more flexible for different datasets
        self.all_labels = sorted(range(0, self.config.n_classes))  # TODO: make this dataset-dependent   

        self.client_loaders_train = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=True) for i, cp in self.client_partitions_train.items() }
        self.client_loaders_test = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=False)  for i, cp in self.client_partitions_test.items() }
        
        self.global_test_loader = DataLoader(self.global_test_partition, batch_size=self.config.global_bs)
    
    def setup_models(self):
        # Create models for each cluster and move to device
        if self.config.model == 'mnist':
            self.models = [MNISTModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]
        if self.config.model == 'resnet':
            self.models = [ResnetModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]        
        if self.config.model == 'femnist':
            self.models = [FemnistModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def global_scheduler(self, round_num, y2=1.0, y1=0.0):
        # simple: every 2 global rounds multiple the lr by a factor of 0.9  
        return self.config.client_lr * (0.9**(round_num//2))
        # simulate one cycle LR
        # return self.config.client_lr * max((1 - math.cos(round_num * math.pi / self.config.global_rounds)) / 2, 0) * (y2 - y1) + y1

    def create_agglomarative_clusters(self, similarities):
        '''
        Create Clustering using Agglomarative clustering
        '''    
        
        distance_matrix = 1 - similarities
        
        aggl_clusterer = AutoAgglomerativeClustering(
            method=self.config.aggl_method,
            metric='precomputed',
            linkage='complete', # mostly will be `complete`
            min_clusters=self.config.min_clusters,  # if clustering based on model state then before starting mostly all models should be in same cluster, then they should start to diverge  
            max_clusters = self.config.max_clusters     
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

    def assign_cluster_model_conv(self, gb_round):
        '''
        assign each of n clients to p clusters
        APPROACH: assign based on the model's state, this will be called every few global rounds to recluster the clients   
        '''
        
        model_states = [model.state_dict() for model in self.models]

        flattend_states = [self.flatten(model_state) for model_state in model_states]

        similarities = self.pairwise_similarity(flattend_states)
        self.similarity_logs[gb_round] = similarities.tolist()
                
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
            result['selected_clients'] = {}
            all_trained_clients = []
            for cl in range(self.config.n_clusters):
                
                if not self.clusters[cl]:  # Skip empty clusters
                    continue            

                # Select clients for this round
                m = max(int(self.config.m * len(self.clusters[cl])), 1)
                selected_clients = np.random.choice(self.clusters[cl], m, replace=False).tolist()
                all_trained_clients.extend(selected_clients)
             
                # Train selected clients
                for client in selected_clients:
                    lr = self.global_scheduler(round_num)
                    train_res = self.train(
                        client, self.models[client], lr, self.client_loaders_train[client]
                    )
                    
                    # dont store the train results, logger becomes too large
                    # result[f'client_{client}'] = train_res
                
                # log the selected clients for this cluster
            result['selected_clients'] = all_trained_clients
            
            is_recluster_round = (self.config.max_clusters > 1 and  round_num % self.config.cluster_every == 0 and  round_num >= self.config.start_recluster)

            # after all clusters are trained then re cluster this round if allowed                        
            if is_recluster_round:
                print(f"Round: {round_num} --- Reclustering")                
                # at this stage each client has been trained on their data for this round not aggregated yet
                self.assign_cluster_model_conv(round_num)


            for cl_id, client_ids_in_cluster in self.clusters.items():
                # Aggregate ONLY the models of clients that were trained this round
                # and belong to the current cluster.
                models_to_aggregate = [
                    self.models[client_id] for client_id in client_ids_in_cluster 
                    if client_id in all_trained_clients
                ]
                if not models_to_aggregate:
                    continue 

                aggregated_model_state = self.cluster_fed_avg(models_to_aggregate)
                
                # Distribute the aggregated model to ALL clients in the cluster
                for client_id in client_ids_in_cluster:
                    self.models[client_id].load_state_dict(aggregated_model_state)
                           
            # Evaluate clients on their test sets if test set exists (independed test set does not exist for femnist)
            # evaluate each client only once every 3 global itertions
            if round_num % self.config.local_eval_every == 0 or round_num == self.config.global_rounds-1:

                avg_acc_selected = []
                avg_acc_all = [] # Will only be populated on full eval rounds
                
                for client_id in range(self.n_clients):
                    if client_id not in self.client_loaders_test:
                        continue

                    # evaluate this client if it was trained this round or if this round all clients are evaluated
                    if client_id in all_trained_clients or ((round_num % 20 == 0 and round_num > 0) or round_num == self.config.global_rounds - 1):
                        client_cluster = -1
                        for cl, clients in self.clusters.items():
                            if client_id in clients:
                                client_cluster = cl
                                break
                        
                        eval_res = self.evaluate(client_id, client_cluster, self.models[client_id], self.client_loaders_test[client_id])
                        
                        # NOTE: Dont store client results as well, logger becomes too large, I might regret this later ....
                        # if f'client_{client_id}' not in result:
                        #     result[f'client_{client_id}'] = {}
                        # result[f'client_{client_id}'].update(**eval_res)

                        if client_id in all_trained_clients : avg_acc_selected.append(eval_res['val_acc'])
                        if client_id in all_trained_clients or ((round_num % 20 == 0 and round_num > 0) or round_num == self.config.global_rounds - 1) : avg_acc_all.append(eval_res['val_acc'])


                if avg_acc_selected:
                    avg_selected = sum(avg_acc_selected) / len(avg_acc_selected)
                    print(f"Average selected clients performance: {avg_selected:.2f}%")
                    result["average_acc_selected"] = avg_selected
                
                if avg_acc_all:
                    avg_all = sum(avg_acc_all) / len(avg_acc_all)
                    print(f"Average all clients performance: {avg_all:.2f}%")
                    result["average_acc_all"] = avg_all

            
            eval_global_res = self.evaluate_global(self.global_test_loader)
            result.update(**eval_global_res)
        
            if self.config.verbose:
                # A more concise way to log cluster assignments
                result['cluster_assignments'] = self.clusters
                print(f"Cluster Assignments: {self.clusters}")
            
            history.append(result)
            self.logger[round_num] = result   

            if round_num > 0 and round_num % 5 == 0:        
                with open(f"checkpoints/{self.config.dataset}_{self.config.n_clients}_clients_{str(int(self.config.m * 100))}_participation_niid_sim_logs_global_eval_round_{round_num}.json", 'w') as f:
                    json.dump({"logs":copy.deepcopy(self.logger),"similarity_logs":self.similarity_logs, "config":asdict(self.config)}, f, default=str)

        return history, self.clusters
