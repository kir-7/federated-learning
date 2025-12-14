import flwr as fl
from flwr.common import ndarrays_to_parameters, Context

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner, PathologicalPartitioner, IidPartitioner

import torch
from dataclasses import dataclass
from torchvision import datasets, transforms
import numpy as np

from ema_strategy import FlowerStrategy
from client import FlowerClient
from models import MNISTModel, CIFAR10Model 
from data_utils import SimpleDataset, FlwrMNISTDataset

@dataclass
class FedConfig:
    n_clients: int = 10
    m: float = 1.0    # fraction of clients of each cluster that needs to be involved in training

    sampling_method: str = 'dirichlet'
    dirichlet_alpha: float = 0.3
    dataset: str = "cifar10"
    n_classes : int = 10
    model : str = 'mnist'

    algorithm : str = 'fedavg'
    prox_lambda : float = 0.25
    k_neighbours : float = 0.4
    ema_alpha : float = 0.75
    num_clusters : int = 3

    client_bs: int = 32
    global_bs: int = 32

    global_rounds: int = 10
    client_epochs: int = 3
    client_lr: float = 1e-4
    reduce_lr_every : int = 5

    train_test_split : float = 0.8

    # seed
    torch_seed: int = 42
    np_seed: int = 43
    rand_seed : int = 44

    init_data_usage : float = 1.0
    data_drift_every : int = 11

    verbose: bool = True
    note: str = "Notes"

config = FedConfig(
    n_clients=10,
    m=1,
    model='cnn',
    
    dataset='cifar10',
    n_classes=10,
    sampling_method='pathological',
    dirichlet_alpha=0.5,
    
    algorithm="ifca",
    k_neighbours = 0.4,
    prox_lambda = 0.001,
    ema_alpha=0.99,
    num_clusters=3,
    
    global_rounds=30,
    client_epochs=3,
    client_bs=64,
    global_bs=128,
    client_lr=1e-3,
    
    init_data_usage=0.5,
    data_drift_every=10,

    verbose=True,

    note="IFCA algorithm, with num clusters=3, using init_data_usage with 0.5 with data being increased every 10 rounds, this doesn't use proximal loss term"
)


torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)

fds = FederatedDataset(
    dataset=config.dataset,
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
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if config.dataset == 'cifar10'  else transforms.Normalize((0.1307, ), (0.3081, ))
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

def client_fn(context : Context):

    cid = str(context.node_config.get("partition-id", context.node_id))

    train_dataset = client_train_partitions[int(cid)]
    test_dataset = client_test_partitions[int(cid)]

    model = CIFAR10Model(n_classes=10)

    return FlowerClient(cid, model, train_dataset, test_dataset, config).to_client()

if __name__ == "__main__":

    # 2. Create Initial Parameters
    initial_net = CIFAR10Model(n_classes=10)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_net.state_dict().items()]
    )

    # 3. Define Strategy
    strategy = FlowerStrategy(
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
        client_resources={"num_cpus": 1, "num_gpus": 1 if torch.cuda.is_available() else 0}
    )