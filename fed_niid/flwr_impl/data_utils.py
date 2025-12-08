import torch

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms

from torchvision.datasets import MNIST, utils
from PIL import Image
import os
import torch
import shutil

# We want to have a way so that we can generate fed dataset for any dataset we want, we can use parititioner from flwr directly and create the dataset ourself


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        super().__init__()

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(x)
        sample = {'img':x, 'label':y}
        return sample

class FlwrMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, client_id, transform=None):
        super().__init__()
        self.client_id = client_id
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_item = self.data[index]

        if isinstance(raw_item, (tuple, list)):
            x, y = raw_item
        else:
            x = raw_item['img']
            y = raw_item['label']
        
        if self.transform:
            x = self.transform(x)
        sample = {'img':x, 'label':y}
        return sample
    
# simple dataset to take hf dataset as input and give img, label as output
class FemnistDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, writer_id, transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.writer_id = writer_id

        self.dataset.set_format('torch')
        self.dataset.set_transform(self.apply_transforms)

    def apply_transforms(self, examples):
        examples["image"] = [self.transform(img) for img in examples["image"]]
        return examples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]
        return {'img': sample['image'], "label": sample['character']}


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
  
    return client_idcs


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


def get_transforms_mnist():
    
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
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    return train_transform, val_transform



if __name__ == '__main__':
    # example to generate client partitions
   
    import torch
    from torchvision import datasets, transforms
    import numpy as np
    from cluster_model_state import FedStateClusterConfig

    torch.manual_seed(42)
    np.random.seed(42)

    config = FedStateClusterConfig(n_clients=10)

    train_data = datasets.EMNIST(root=".", split="byclass", download=True, train=True)
    test_data = datasets.EMNIST(root=".", split="byclass", download=True, train=False)

    idcs = np.random.permutation(len(train_data))    

    all_cleint_idcs = idcs[:config.total_train_samples]

    per_client_samples = len(all_cleint_idcs) // config.n_clients

    train_labels = train_data.train_labels.numpy()

    client_idcs = split_noniid(all_cleint_idcs, train_labels, alpha=config.dirichlet_alpha, n_clients=config.n_clients)

    client_data = [Subset(train_data, idcs) for idcs in client_idcs]    

    client_train_partitions, client_test_partitions = {}, {}

    for i in range(config.n_clients):        
        n_train= int(config.train_test_split*len(client_data[i]))
        n_eval =  len(client_data[i]) - n_train
        client_i_train, client_i_test = torch.utils.data.random_split(client_data[i], [n_train, n_eval])
        
        # to create rotated MNIST
        if i < (0.5*config.n_clients):
            client_train_partitions[i] = SimpleDataset(client_i_train, transforms.Compose([transforms.RandomRotation((180,180)),
                                                        transforms.ToTensor()]))
        else:        
            client_train_partitions[i] = SimpleDataset(client_i_train, transforms.Compose([transforms.ToTensor()]))
        
        client_test_partitions[i] = SimpleDataset(client_i_test, transforms.Compose([transforms.ToTensor()]))        
     
      

    global_test_data = SimpleDataset(Subset(test_data, np.arange(config.total_test_samples)), transforms.Compose([transforms.ToTensor()]))            
    

