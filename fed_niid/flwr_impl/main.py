import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Code,
)
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner, PathologicalPartitioner, IidPartitioner

import torch
from dataclasses import dataclass
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
import os

from models import MNISTModel, CIFAR10Model # Assuming your models.py is available
from client import FlowerClient
from strategy import FedGraphStrategy
from data_utils import SimpleDataset, FlwrMNISTDataset

@dataclass
class FedGraphConfig:
    n_clients: int = 10
    m: float = 1.0    # fraction of clients of each cluster that needs to be involved in training
    
    method: str = 'dirichlet'
    dirichlet_alpha: float = 0.3
    dataset: str = "cifar10"    
    n_classes : int = 10
    model : str = 'mnist'
    
    algorithm : str = 'fed_g_prox'
    prox_lambda : float = 0.25
    k_neighbours : float = 0.4
    ema_alpha : float = 0.75 

    n_communities : int = 3

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
    start_graph : int = 10
    swap_dist_every : int = 8

    log_dir : str = "checkpoints"
    verbose: bool = True

config = FedGraphConfig(
    n_clients=15,
    n_classes=10,
    client_lr=1e-3,
    dataset='cifar10',
    method='pathological',

    k_neighbours = 1.0,
    prox_lambda = 0.001,

    dirichlet_alpha=0.5,
    model='cnn',
    m=1,
    global_rounds=50,
    client_epochs=3,
    client_bs=64,
    global_bs=128,

    # total_train_samples= 300000,
    # total_test_samples = 60000,

    local_eval_every = 1,
    swap_dist_every = 51,
    start_graph = 10,
    log_dir='checkpoints_3'
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CPUS_PER_CLIENT = 1
NUM_GPUS_PER_CLIENT = 0.5 if torch.cuda.is_available() else 0

torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)

os.makedirs(f"/content/{config.log_dir}", exist_ok=True)

# load full training data using flwr and split it int train and test

fds = FederatedDataset(
    dataset="cifar10",
    partitioners={
        "train": PathologicalPartitioner(
            partition_by="label",
            num_partitions=config.n_clients,
            num_classes_per_partition=4,
            class_assignment_mode="first-deterministic"
        )
    }
)

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

client_train_partitions = {}
client_test_partitions = {}

print("Splitting data for clients...")
for client_id in range(config.n_clients):
    partition = fds.load_partition(partition_id=client_id, split="train")
    
    split_dataset = partition.train_test_split(train_size=config.train_test_split, seed=config.np_seed)

    client_train_partitions[client_id] = FlwrMNISTDataset(
        split_dataset['train'], client_id, transform=cifar_transform
    )
    client_test_partitions[client_id] = FlwrMNISTDataset(
        split_dataset['test'], client_id, transform=cifar_transform
    )

global_train_loaders = { 
    i: DataLoader(ds, batch_size=config.client_bs, shuffle=True) 
    for i, ds in client_train_partitions.items() 
}
global_test_loaders = { 
    i: DataLoader(ds, batch_size=config.client_bs, shuffle=False)  
    for i, ds in client_test_partitions.items() 
}
    
def client_fn(cid: str):
    # Load model and data specific to this client ID
    # Note: You need to adapt your 'client_partitions' logic to be accessible here
    # Since client_fn only takes 'cid', you usually load data globally or inside here.
    
    # Example assuming you have global access to your data partitions
    # In production, pass a data_loader_factory
    
    train_loader = global_train_loaders[int(cid)]
    test_loader = global_test_loaders[int(cid)]
    
    model = CIFAR10Model(n_classes=10)
    
    return FlowerClient(cid, model, train_loader, test_loader, config)

if __name__ == "__main__":

    # 2. Create Initial Parameters
    initial_net = CIFAR10Model(n_classes=10)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_net.state_dict().items()]
    )

    # 3. Define Strategy
    strategy = FedGraphStrategy(
        num_clients=config.n_clients,
        initial_parameters=initial_params,
        k_neighbours=config.k_neighbours,
        fraction_fit=config.m,
    )

    # 4. Start Simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.n_clients,
        config=fl.server.ServerConfig(num_rounds=config.global_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.5 if torch.cuda.is_available() else 0}
    )