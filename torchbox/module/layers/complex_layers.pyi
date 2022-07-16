"""

import torch as th
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv2d, Conv1d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d, ConvTranspose1d
from torch.nn import Upsample
from torchbox.layerfunction.complex_functions import complex_relu, complex_leaky_relu, complex_max_pool2d, complex_max_pool1d
from torchbox.layerfunction.complex_functions import complex_dropout, complex_dropout2d
from torchbox.layerfunction.cplxfunc import csoftshrink, softshrink


class ComplexSoftShrink(Module):
    def __init__(self, alpha=0.5, cdim=None, inplace=False):
    def forward(self, input, alpha=None):
class SoftShrink(Module):
    def __init__(self, alpha=0.5, inplace=False):
    def forward(self, input, alpha=None):
class ComplexSequential(Sequential):
    def forward(self, input):
class ComplexDropout(Module):
    def __init__(self, p=0.5, inplace=False):
    def forward(self, input):
class ComplexDropout2d(Module):
    def __init__(self, p=0.5, inplace=False):
    def forward(self, input):
class ComplexMaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0,
    def forward(self, input):
class ComplexMaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0,
    def forward(self, input):
class ComplexReLU(Module):
    def __init__(self, inplace=False):
    def forward(self, input):
class ComplexLeakyReLU(Module):
    def __init__(self, negative_slope=(0.01, 0.01), inplace=False):
    def forward(self, input):
class ComplexConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
    def forward(self, input):
class ComplexConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, input):
class ComplexConvTranspose1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
    def forward(self, input):
class ComplexConv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, input):
class ComplexUpsample(Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
    def forward(self, input):
class ComplexLinear(Module):
    def __init__(self, in_features, out_features):
    def forward(self, input):
class NaiveComplexBatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
    def forward(self, input):
class NaiveComplexBatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
    def forward(self, input):
class _ComplexBatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
    def reset_running_stats(self):
    def reset_parameters(self):
class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, input):
class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, input):
class ComplexConv1(Module):
    def __init__(self, axis, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, Xr, Xi):
class ComplexMaxPool1(Module):
    def __init__(self, axis, kernel_size, stride=None, padding=0,
    def forward(self, Xr, Xi):
class ComplexConv2(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, Xr, Xi):
class ComplexMaxPool2(Module):
    def __init__(self, kernel_size, stride=None, padding=0,
    def forward(self, Xr, Xi):
