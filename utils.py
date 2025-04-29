import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from smapling import cancer_dataset, breast_cancer_iid, breast_cancer_noniid
from dataclasses import dataclass


@dataclass
class ARGS:
    model : str = "mlp"
    optimizer : str = 'adam'
    num_channels : int = 1
    kernel_num : int  = 9
    kernel_size : tuple = (3,4, 5)
    norm : str = "batch_norm"
    num_filters:int = 32
    max_pool : bool = True
    lr : str = 1e-2
    epochs : int = 10
    iid : bool = True
    frac : float = 0.1
    num_users : int = 100
    local_bs: int = 10
    local_ep : int = 10
    dataset : str = 'cancer'
    num_classes :int=10
    gpu : bool = False
    verbose=False
    momentum : float = 0.5
    unequal:int = 0 
    seed : int = 1 
    n_clients:int=100

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cancer':
        

        # data transformation 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        composed_train = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(degrees=5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

        # this transformation is for valiadationa and test sets
        composed= transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

        data_dir = data_dir = "./"
        dataset_full = cancer_dataset(data_dir, transform=composed)
        
        # Create a subset with just the first 4000 samples using slicing
        dataset = dataset_full[:4000]

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = breast_cancer_iid(dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = breast_cancer_noniid(dataset, args.num_users)
                
    
    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return user_groups


def average_weights(w):    #rename to fed_avg
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg

@torch.no_grad()
def calc_accuracy(model, criterion, test_loader, device):
    
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    loss_avg = loss/len(test_loader)

    accuracy = correct/total
    return accuracy, loss_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
