import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from tqdm.auto import tqdm
from models import MNISTModel, ResnetModel, FemnistModel, CIFAR10Model

import copy
import json

@dataclass
class FedGraphConfig:
    n_clients: int = 10
    m: float = 1.0    # fraction of clients of each cluster that needs to be involved in training
    
    method: str = 'dirichlet'
    dirichlet_alpha: float = 0.3
    dataset: str = "cifar10"    
    n_classes : int = 10
    model : str = 'mnist'
    prox_lambda : float = 0.25
    k_neighbours : float = 0.4 

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
    swap_dist_every : int = 8

    log_dir : str = "checkpoints"
    verbose: bool = True


class FedGraph:
    def __init__(self, config: FedGraphConfig, global_test_partition:dict[str, torch.utils.data.Dataset], client_parition_train:dict[str, torch.utils.data.Dataset], client_partitions_test:dict[str, torch.utils.data.Dataset]):
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


    def setup_dataset(self):        
        
        # Make this more flexible for different datasets
        self.all_labels = sorted(range(0, self.config.n_classes))  # TODO: make this dataset-dependent   

        self.client_loaders_train = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=True) for i, cp in self.client_partitions_train.items() }
        self.client_loaders_test = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=False)  for i, cp in self.client_partitions_test.items() }
        
        self.global_test_loader = DataLoader(self.global_test_partition, batch_size=self.config.global_bs)
    
    def setup_models(self):
        # Create models for each client and move to device
        if self.config.model == 'mnist':
            self.models = [MNISTModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]
        if self.config.model == 'resnet':
            self.models = [ResnetModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]        
        if self.config.model == 'femnist':
            self.models = [FemnistModel(n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]
        if self.config.model == 'cnn':
            self.models = [CIFAR10Model(in_channels=3, n_classes=self.config.n_classes).to(self.device) for _ in range(self.config.n_clients)]
        
        self.cross_entropy = nn.CrossEntropyLoss().to(self.device)

    def criterion(self, model, prev_model_state, y_pred, y_true):
        # do something similar to fedprox

        loss_ce = self.cross_entropy(y_pred, y_true)
        
        loss_prox = 0.0
                
        for prev_param, new_param in zip(prev_model_state.parameters(), model.parameters()):
            loss_prox += torch.sum((new_param - prev_param) ** 2)
        
        return loss_ce + self.config.prox_lambda * loss_prox


    def global_scheduler(self, round_num, y2=1.0, y1=0.0):
        # simple: every 2 global rounds multiple the lr by a factor of 0.9  
        return self.config.client_lr * (0.9**(round_num//2))
        # simulate one cycle LR
        # return self.config.client_lr * max((1 - math.cos(round_num * math.pi / self.config.global_rounds)) / 2, 0) * (y2 - y1) + y1

    @staticmethod
    def flatten(source):
        return torch.cat([value.flatten() for value in source])
    
    def build_graph(self):
        # setup client graph

        # use an adjecency matrix 
        model_states = torch.stack([self.flatten(model.parameters()).detach() for model in self.models])
        distance_mat = torch.cdist(model_states, model_states, p=2).detach()  # shape: N x D
                
        # better to have graph made of similarities, sim = e^(-dist)
        mean_dist = distance_mat.mean() 
        sigma = mean_dist if mean_dist > 0 else 1.0
        self.graph = torch.exp(-distance_mat/sigma)

        # reverse-deleting
        # prune n*(n-1)/4 least weighted edges ( prune edges with less similarity score )
        k = max(2, int(self.n_clients * self.config.k_neighbours))
        vals, indices = torch.topk(self.graph, k=k, dim=1)
        
        new_graph = torch.zeros_like(self.graph)
        new_graph.scatter_(1, indices, vals)
        row_sums = new_graph.sum(dim=1, keepdim=True)
        self.graph = new_graph / (row_sums + 1e-8)

    def final_cluster(self):
        # after training is done, perform community detection and get the communities, these will be the final cluster graphs
        pass


    def fed_avg(self):                
        wts_dict = [{name:param.data for name, param in model.named_parameters()} for model in self.models]

        # self.graph is NxN and wts_dict is NxD
        avg_wts_all_models = {}
        
        for client_id in range(self.n_clients):
            client_agg_weights = {}
            
            # Get the row of weights for this client: Shape [N_clients]
            neighbor_weights = self.graph[client_id] 
            
            for name in wts_dict[0].keys():
                # stack params: shape [n_clients, *param_shape] [10, 64, 3, 3]
                stacked_params = torch.stack([wt[name] for wt in wts_dict], dim=0)
                
                view_shape = [-1] + [1] * (stacked_params.dim() - 1)
                reshaped_weights = neighbor_weights.view(view_shape)              
               
                weighted_sum = torch.sum(stacked_params * reshaped_weights, dim=0)
                
                client_agg_weights[name] = weighted_sum
       
        # return the aggregated weights
        return avg_wts_all_models        

    @torch.inference_mode()
    def evaluate(self, client, model, dataloader):
        model.eval()
        
        correct = 0
        total = 0
        test_loss = 0
        
        for sample in tqdm(dataloader, desc=f"Validating clinet: {client}", leave=False):
            images, labels = sample['img'], sample['label']
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            test_loss += self.cross_entropy(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = test_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        return {"val_acc": accuracy, "val_avg_loss": avg_loss}
    
    def train(self, client, prev_model_state, model, lr, dataloader):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.lr_schd_gamma)
        
        epoch_loss = []
        
        for epoch in tqdm(range(self.config.client_epochs), desc=f"Training clinet: {client}", leave=False):
            batch_loss = []
            
            for batch_idx, sample in enumerate(dataloader):
                
                images, labels = sample['img'], sample['label']
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()  
                outputs = model(images)
                loss = self.criterion(model, prev_model_state, outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            
            # scheduler.step()
            
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
                
            
            # Select clients for this round
            m = max(int(self.config.m * self.n_clients), 1)
            selected_clients = np.random.choice(range(self.config.n_clients), m, replace=False).tolist()
            all_trained_clients.extend(selected_clients)
            
            # Train selected clients
            for client in selected_clients:
                lr = self.global_scheduler(round_num)
                train_res = self.train(
                    client, copy.deepcopy(self.models[client]), self.models[client], lr, self.client_loaders_train[client]
                )
                            
            result['selected_clients'] = all_trained_clients                                                                
           
            # before federation, refine the graph
            self.build_graph()

            aggregated_wts_all_models = self.fed_avg()
                
            for client_id, agg_wts in aggregated_wts_all_models.items():
                    for name, param in self.models[client_id].named_parameters(): 
                        if param.requires_grad:
                            param.data.copy_(agg_wts[name])
                           
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
                        
                        eval_res = self.evaluate(client_id, self.models[client_id], self.client_loaders_test[client_id])
                        
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

            
                    
            # once every few rounds, swap the distributions of any 2 random clients, to simulate data drift
            if round_num > 0 and round_num % self.config.swap_dist_every == 0:
                # swap distribution of few clients
                for _ in range(int((self.n_clients // 2)*0.4)+1):
                    r_client_1, r_client_2 = np.random.choice(a=self.config.n_clients, size=2, replace=False)
                    
                    self.client_partitions_train[r_client_1], self.client_partitions_train[r_client_2] = self.client_partitions_train[r_client_2], self.client_partitions_train[r_client_1]
                    self.client_loaders_train[r_client_1], self.client_loaders_train[r_client_2] = self.client_loaders_train[r_client_2], self.client_loaders_train[r_client_1]
                    
                    self.client_partitions_test[r_client_1], self.client_partitions_test[r_client_2] = self.client_partitions_test[r_client_2], self.client_partitions_test[r_client_1]
                    self.client_loaders_test[r_client_1], self.client_loaders_test[r_client_2] = self.client_loaders_test[r_client_2], self.client_loaders_test[r_client_1]
                                    
                    print(f"Swapped distributions of clients: {r_client_1} & {r_client_2}")

            if self.config.verbose and round_num % 5 == 0:
                self.plot_graph(round_num)

            history.append(result)
            self.logger[round_num] = result   

            if (round_num > 0 and round_num % 5 == 0) or round_num == self.config.global_rounds - 1:        
                with open(f"{self.config.log_dir}/{self.config.dataset}_{self.config.n_clients}_clients_{str(int(self.config.m * 100))}_participation_niid_sim_logs_global_eval_round_{round_num}.json", 'w') as f:                    
                    json.dump({"logs":copy.deepcopy(self.logger),"similarity_logs":self.similarity_logs, "config":asdict(self.config)}, f, default=str)
        
        return history    


    def plot_graph(self, round_num):    
        plt.figure(figsize=(8, 6))
        rows, cols = np.where(self.graph.detach().cpu().numpy() > 0)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.write_graphml(gr, f"{self.config.log_dir}/graphml_{self.config.dataset}_{self.config.n_clients}_clients_{str(int(self.config.m * 100))}_participation_round_{round_num}.graphml")      
        nx.draw(gr, node_size=500, with_labels=True)
        plt.savefig(f"{self.config.log_dir}/{self.config.dataset}_{self.config.n_clients}_clients_{str(int(self.config.m * 100))}_participation_round_{round_num}.png")
        plt.close()
