import torch

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms

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
        return x, y
    

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

    # mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U1')

    mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
        'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
        'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], dtype='<U1')

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
    

