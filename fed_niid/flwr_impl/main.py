import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Code,
    Context
)

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner, PathologicalPartitioner, IidPartitioner

import torch
from dataclasses import dataclass
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset
import random

from avg_strategy import FlowerStrategy
from data_drift_client import FlowerClient

from models import MNISTModel, CIFAR10Model
from data_utils import SimpleDataset, FlwrMNISTDataset

@dataclass
class FedConfig:
    n_clients: int = 10
    m: float = 1.0    # fraction of clients of each cluster that needs to be involved in training

    sampling_method: str = 'dirichlet'
    dirichlet_alpha: float = 0.3
    num_classes_per_partition:int = 3
    dataset: str = "cifar10"
    partition_column:str = 'label'
    n_classes : int = 10
    model : str = 'mnist'
    k_clusters : int = 3

    algorithm : str = 'fed_g_prox'
    prox_lambda : float = 0.25
    k_neighbours : float = 0.4
    ema_alpha : float = 0.75

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

    init_data_usage : float = 0.5
    data_drift_every : int = 15

    verbose: bool = True
    note: str = "Notes"

config = FedConfig(
    n_clients=40,
    n_classes=10,
    client_lr=1e-3,
    dataset='cifar10',
    partition_column='label',
    sampling_method='pathological',

    k_neighbours = 0.4,
    k_clusters=4,
    prox_lambda = 0,
    ema_alpha=0.99,
    algorithm="fed_sim_cl",

    dirichlet_alpha=0.5,
    num_classes_per_partition=3,
    model='cnn',
    m=0.3,
    global_rounds=50,
    client_epochs=4,
    client_bs=64,
    global_bs=128,

    init_data_usage=1.0,
    data_drift_every=31,

    verbose=True,

    note="experiment 1.4 of varying NonIIDness: fed Sim CL algorithm, 40 clients and 30% participation and pathelogical sampling with num_classes_per_partition=3 and k_clusters=4 and NO data drift. evaluated every 5 rounds"
)

torch.manual_seed(config.torch_seed)
np.random.seed(config.np_seed)
random.seed(config.rand_seed)

fds = FederatedDataset(
    dataset=config.dataset,
    partitioners={
        "train": PathologicalPartitioner(
            partition_by=config.partition_column,
            num_partitions=config.n_clients,
            num_classes_per_partition=config.num_classes_per_partition,
            class_assignment_mode="first-deterministic",
            seed=config.rand_seed
        )
    }
)

if config.dataset == 'cifar100' or config.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
elif config.dataset == 'mnist':
    mean = (0.1307, )
    std = (0.3081, )

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

client_train_partitions = {}
client_test_partitions = {}

print("Splitting data for clients...")
for client_id in range(config.n_clients):
    partition = fds.load_partition(partition_id=client_id, split="train")

    split_dataset = partition.train_test_split(train_size=config.train_test_split, seed=config.np_seed)

    train_dataset = FlwrMNISTDataset(
        split_dataset['train'], client_id, transform=cifar_transform
    )

    val_dataset = FlwrMNISTDataset(
        split_dataset['test'], client_id, transform=cifar_transform
    )

    train_labels = [sample['label'] for sample in train_dataset]
    val_labels = [sample['label'] for sample in val_dataset]

    train_sorted = torch.argsort(torch.Tensor(train_labels)).tolist()
    val_sorted = torch.argsort(torch.Tensor(val_labels)).tolist()

    sorted_train_dataset = Subset(train_dataset, train_sorted)
    sorted_val_dataset = Subset(val_dataset, val_sorted)

    client_train_partitions[client_id] = sorted_train_dataset
    client_test_partitions[client_id] = sorted_val_dataset


def client_fn(context : Context):

    cid = str(context.node_config.get("partition-id", context.node_id))

    train_dataset = client_train_partitions[int(cid)]
    test_dataset = client_test_partitions[int(cid)]

    model = CIFAR10Model(n_classes=config.n_classes)

    return FlowerClient(cid, model, train_dataset, test_dataset, config).to_client()


if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CPUS_PER_CLIENT = 1
    NUM_GPUS_PER_CLIENT = 0.15

    initial_net = CIFAR10Model(n_classes=10)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_net.state_dict().items()]
    )

    strategy = FlowerStrategy(
        num_clients=config.n_clients,
        initial_parameters=initial_params,
        fraction_fit=config.m,
        fraction_evaluate=1.0,
        evaluate_frequency=5,
        total_rounds=config.global_rounds,
        # start_knn=5,
        # k_neighbours=config.k_neighbours
        k_clusters=config.k_clusters
    )


    history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=config.n_clients,
    config=fl.server.ServerConfig(num_rounds=config.global_rounds),
    strategy=strategy,
    client_resources={"num_cpus": NUM_CPUS_PER_CLIENT, "num_gpus":NUM_GPUS_PER_CLIENT},
    ray_init_args={"log_to_driver": True, "include_dashboard": False}
)