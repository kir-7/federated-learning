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


class FEMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.download = download
        self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
        self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'
        self.train = train
        self.root = root
        self.training_file = f'{self.root}/FEMNIST/processed/femnist_train.pt'
        self.test_file = f'{self.root}/FEMNIST/processed/femnist_test.pt'
        self.user_list = f'{self.root}/FEMNIST/processed/femnist_user_keys.pt'

        if not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_test.pt') \
                or not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_train.pt'):
            if self.download:
                self.dataset_download()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data_targets_users = torch.load(data_file)
        self.data, self.targets, self.users = torch.Tensor(data_targets_users[0]), torch.Tensor(data_targets_users[1]), data_targets_users[2]
        self.user_ids = torch.load(self.user_list)

    def __getitem__(self, index):
        img, target, user = self.data[index], int(self.targets[index]), self.users[index]
        img = Image.fromarray(img.numpy(), mode='F')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, user

    def dataset_download(self):
        paths = [f'{self.root}/FEMNIST/raw/', f'{self.root}/FEMNIST/processed/']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # download files
        filename = self.download_link.split('/')[-1]
        utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}/FEMNIST/raw/', filename=filename, md5=self.file_md5)

        files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
        for file in files:
            # move to processed dir
            shutil.move(os.path.join(f'{self.root}/FEMNIST/raw/', file), f'{self.root}/FEMNIST/processed/')

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
    

