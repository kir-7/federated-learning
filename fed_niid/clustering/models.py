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
    
    def forward(self, x):
        return self.model(x)    


class CIFAR10Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super().__init__()

        self.n_classes = n_classes

        # self.blocks = nn.ModuleList([
        #     nn.Sequential(*[
        #         Conv(in_channels, 32, 3, 2),    # /2
        #         Conv(32, 64, 3, 2),    # /4
        #         Conv(64, 128, 3, 1),   # /4
        #     ]),
        #     nn.Sequential(*[
        #         Conv(128, 128, 3, 2),    # /8
        #         Conv(128, 64, 3, 1),    # /8
        #         Conv(64, 64, 3, 1),     # /8                
        #     ])
        # ])

        self.blocks = nn.Sequential(Conv(3, 32, 3, 2), Conv(32, 64, 3, 2), Conv(64, 128, 3, 1), Conv(128, 64, 3, 2))

        self.pool = nn.AdaptiveAvgPool2d(1)        

        self.fc = nn.Sequential(nn.Flatten(1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, x):

        # for block in self.blocks:
        #     x = block(x)        

        x = self.pool(x)

        return self.fc(x)


class MNISTModel(nn.Module):
    def __init__(self, in_features=28*28, n_classes=10):
        super().__init__()

        self.n_classes = n_classes

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes)) 

    def forward(self, x):
        return self.fc(x)   