'''
Check if models with similar data converge to similar locations using the FEMNIST Dataset.
'''

import random
import torch
from torch import nn

from torchvision import transforms

from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from dataclasses import dataclass

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

from models import MNISTModel, ResnetModel

@dataclass
class FedConvergeConfig:
    n_clients: int = 10
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
       
    verbose: bool = True

class FedConverge:
    def __init__(self, config:FedConvergeConfig):  

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(self.config.torch_seed)        
        np.random.seed(self.config.np_seed)
        
        self.n_clients = config.n_clients
        self.global_rounds = config.global_rounds
        
        if config.dataset == 'cifar10':
            self.train_transform, self.simple_train_transform, self.val_transform = self.get_transforms_cifar10()
        elif config.dataset == "mnist":
            self.train_transform, self.simple_train_transform, self.val_transform = self.get_transforms_mnist()
        elif config.dataset == 'rotmnist':
            self.train_transform, self.simple_train_transform, self.val_transform = self.get_transforms_rotmnist()
        else:
            raise ValueError(f"dataset: {config.dataset} not allowed")

        # will store all the client models states across all global epochs: for each client the value
        # will be a dict for each round starting from 0 to config.global_rounds both included
        self.local_model_states = {client: {} for client in range(self.config.n_clients)}
        self.gloabl_model_states = {}
        
        self.setup_dataset()
        self.setup_models()
    
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

        simple_train_transform = transforms.Compose([
            # transforms.RandomRotation(10), 
            transforms.ToTensor(),         
            transforms.Normalize(0.1307, 0.3081) # Normalizes the tensor
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),         
            transforms.Normalize(0.1307, 0.3081) # Normalizes the tensor
        ])

        return train_transform, simple_train_transform, val_transform

    def get_transforms_cifar10(self):
        
        train_transform = transforms.Compose([
            transforms.ToTensor(),            
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) if self.config.dataset == 'cifar10' else transforms.Normalize((0.1307,), (0.3081,))
        ])
        simple_train_transform = transforms.Compose([
            transforms.ToTensor(),            
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) if self.config.dataset == 'cifar10' else transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  if self.config.dataset == 'cifar10' else transforms.Normalize((0.1307,), (0.3081,))
        ])

        return train_transform, simple_train_transform, val_transform

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
                    self.client_partitions_test[i] =  fds.load_partition(i, "test").with_transform(self.apply_transforms_train)
       
            # Load the global test partition for other datasets
            self.global_test_partition = fds.load_split('test').with_transform(self.apply_transforms_test)
        
        # Make this more flexible for different datasets
        self.all_labels = sorted(range(0, self.config.n_classes))  # TODO: make this dataset-dependent   

        self.client_loaders_train = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=True) for i, cp in self.client_partitions_train.items() }
        self.client_loaders_test = { i: DataLoader(cp, batch_size=self.config.client_bs, shuffle=False)  for i, cp in self.client_partitions_test.items() }
        
        self.global_test_loader = DataLoader(self.global_test_partition, batch_size=self.config.global_bs)
    

    def setup_models(self):

        if self.config.dataset == 'mnist' or self.config.dataset == 'flwrlabs/femnist' or self.config.dataset == 'rotmnist':
            self.model = MNISTModel(n_classes=self.config.n_classes).to(self.device)
        else:
            self.model = ResnetModel(n_classes=self.config.n_classes).to(self.device)
        
        self.clients = list(range(self.config.n_clients))

        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
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
    def evaluate(self, client, model, dataloader):
        model.eval()
        
        correct = 0
        total = 0
        test_loss = 0
        
        for sample in tqdm(dataloader, desc=f"Validating clinet: {client}", leave=False):
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
            
            # CORRECT PLACEMENT: Called once per epoch
            scheduler.step()
            
            avg_loss = sum(batch_loss) / len(batch_loss) if batch_loss else 0
            epoch_loss.append(avg_loss)
        
        return model, {
            'avg_train_loss': sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0,
            "train_loss": epoch_loss[-1] if epoch_loss else 0
        }
    
    def run(self):        

        history = []

        # store the initial state
        for client in self.clients:
            self.local_model_states[client][0] = deepcopy(self.model)
        
        for round_num in range(self.config.global_rounds):
            
            if self.config.verbose:
                print(f"\n=== Global Round {round_num + 1}/{self.config.global_rounds} ===")
            
            result = {}
            
            # select clients for this round
            m = max(int(self.config.m * self.config.n_clients), 1)
            selected_clients = np.random.choice(self.clients, m, replace=False).tolist()            
            
            local_models = []

            for client in selected_clients:
                local_trained_model, train_res = self.train(
                    client,
                    deepcopy(self.model), 
                    self.client_loaders_train[client]
                )

                local_models.append(local_trained_model)
                
                
                self.local_model_states[client][round_num+1] = local_trained_model
                

                result[f'client_{client}'] = train_res
            
            # Aggregate models
            self.cluster_fed_avg(local_models, self.model)    

            self.gloabl_model_states[round_num+1] = deepcopy(self.model)

            if round_num % 5 == 0:
                    for client in selected_clients:
                        if client in self.client_loaders_test: 
                            eval_res = self.evaluate(client, self.model, self.client_loaders_test[client])
                            result[f'client_{client}'].update(**eval_res)

            gb_eval_res = self.evaluate('Aggregated', self.model, self.global_test_loader)

            result.update(**gb_eval_res)

            if self.config.verbose:
                print(result)            
                print(f"Global Round:{round_num}  Val Accuracy: {gb_eval_res['val_acc']}")
            history.append(result)
        
        return history
