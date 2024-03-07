from torch import nn as nn
import torch
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass

def conv2d(ch_in, ch_out, kz, s=1, p=0):
    return spectral_norm(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kz, stride=s, padding=p))

def block1():
    return seq(
        conv2d(ch_in=3, ch_out=64, kz=3, s=1, p=1), nn.ReLU(),
        conv2d(ch_in=64, ch_out=64, kz=3, s=1, p=1), nn.ReLU()
    )

def block2():
    return seq(
        conv2d(ch_in=64, ch_out=128, kz=3, s=1, p=1), nn.ReLU(),
        conv2d(ch_in=128, ch_out=128, kz=3, s=1, p=1), nn.ReLU()
    )

def block3():
    return seq(
        conv2d(ch_in=128, ch_out=256, kz=3, s=1, p=1), nn.ReLU(),
        conv2d(ch_in=256, ch_out=256, kz=3, s=1, p=1), nn.ReLU(),
        conv2d(ch_in=256, ch_out=256, kz=1, s=1, p=0), nn.ReLU()
    )

def block4():
    return seq(
        conv2d(ch_in=256, ch_out=512, kz=3, s=1, p=1), nn.ReLU(),
        conv2d(ch_in=512, ch_out=512, kz=3, s=1, p=1), nn.ReLU(),
        conv2d(ch_in=512, ch_out=512, kz=1, s=1, p=0), nn.ReLU()
    )

def block5():
    return seq(
        conv2d(ch_in=512, ch_out=512, kz=3, s=1, p=1), nn.ReLU(),
        conv2d(ch_in=512, ch_out=512, kz=3, s=1, p=1), nn.ReLU(),
        conv2d(ch_in=512, ch_out=512, kz=1, s=1, p=0), nn.ReLU()
    )

def fc_layer(num_class):
    return seq(
        nn.Linear(in_features=7*7*512, out_features=4096), nn.ReLU(),
        nn.Linear(in_features=4096, out_features=1000), nn.ReLU(),
        nn.Linear(in_features=1000, out_features=num_class), nn.ReLU(),
    )

class VGG16(nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()

        self.block = seq(
            block1(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            block2(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            block3(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            block4(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            block5(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            fc_layer(num_class=num_class), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.block(x)