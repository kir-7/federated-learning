import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, d=1, act=nn.ReLU):
        super().__init__()

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=autopad(k, p, d), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, x:torch.Tensor):
        return self.act(self.bn(self.conv(x)))

class ResnetModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = resnet18(num_classes=n_classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)    
    
class FemnistModel(torch.nn.Module):
    def __init__(self, in_features=1, n_classes=62):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            Conv(c_in=in_features, c_out=32, k=5),
            Conv(32, 32, k=3, s=2),   # 14x14
            Conv(32, 64, 3),
            Conv(64, 64, 3, 2),  # 7x7   
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

class CIFAR10Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super().__init__()

        self.n_classes = n_classes

        self.conv_layers = nn.Sequential(
            Conv(c_in=in_channels, c_out=32, k=5),
            Conv(32, 32, k=3, s=2),   # 16x16
            Conv(32, 64, 3),
            Conv(64, 64, 3, 2),  # 8x8
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
       
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class MNISTModel(nn.Module):
    def __init__(self, in_features=28*28, n_classes=10):
        super().__init__()

        self.n_classes = n_classes

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes)) 

    def forward(self, x):
        return self.fc(x)   