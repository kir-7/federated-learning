
import tprch
from torch import nn
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, Linear, global_mean_pool
from torch_geometric.nn.inits import zeros, glorot, reset
from torch_geometric.nn.models import MLP
from torch_geometric.utils import add_self_loops, degree, get_laplacian, remove_self_loops, spmm


'''
We want to compress a model's state and represent it as a low dim vector using a GNN

Input is a model's state (Graph) and output is a vector representing the model's state

compress the model's parameters to reduce the nodes in the GNN

'''

