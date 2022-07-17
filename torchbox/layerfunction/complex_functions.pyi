"""

import torch as th
from torch.nn.functional import relu, leaky_relu, max_pool2d, max_pool1d, dropout, dropout2d, upsample


def complex_relu(input, inplace=False):
    ...

def complex_leaky_relu(input, negative_slope=(0.01, 0.01), inplace=False):
    ...

def complex_max_pool2d(input, kernel_size, stride=None, padding=0,    ...

def complex_max_pool1d(input, kernel_size, stride=None, padding=0,    ...

def complex_dropout(input, p=0.5, training=True, inplace=False):
    ...

def complex_dropout2d(input, p=0.5, training=True, inplace=False):
    ...

def complex_upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    ...

