#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : convolution.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 0.0
# @license   : The GNU General Public License (GPL) v3.0
# @note      : 
# 
# The GNU General Public License (GPL) v3.0
# Copyright (C) 2013- Zhi Liu
#
# This file is part of torchbox.
#
# torchbox is free software: you can redistribute it and/or modify it under the 
# terms of the GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or (at your option) any later version.
#
# torchbox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with torchbox. 
# If not, see <https://www.gnu.org/licenses/>. 
#

import torch as th
from torch.nn.parameter import Parameter
from torchbox.dsp.convolution import fftconv1


class FFTConv1(th.nn.Module):

    def __init__(self, nh, h=None, shape='same', nfft=None, train=True, **kwargs):
        super(FFTConv1, self).__init__()
        if 'cdim' in kwargs:
            self.cdim = kwargs['cdim']
        elif 'caxis' in kwargs:
            self.cdim = kwargs['caxis']
        else:
            self.cdim = None

        if 'dim' in kwargs:
            self.dim = kwargs['dim']
        elif 'axis' in kwargs:
            self.dim = kwargs['axis']
        else:
            self.dim = 0

        self.nfft = nfft
        self.shape = shape
        if h is None:
            self.h = Parameter(th.randn(nh + [2]), requires_grad=train)
        else:
            self.h = Parameter(h, requires_grad=train)

    def forward(self, x):
        y = fftconv1(x, self.h, shape=self.shape, cdim=self.cdim, dim=self.dim, nfft=self.nfft,
                     ftshift=False, eps=None)
        return y


class Conv1(th.nn.Module):

    def __init__(self, axis, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv1, self).__init__()
        if axis not in [2, 3]:
            raise ValueError('Only support 2 or 3 for N-C-H-W-2')
        self.axis = axis
        self.conv = th.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, X):
        N, C, H, W = X.size()
        axis = 5 - self.axis
        X = X.permute(0, axis, 1, self.axis)
        size = list(X.size())
        X = X.contiguous().view(size[0] * size[1], size[2], size[3])
        # X = X.reshape(size[0] * size[1], size[2], size[3])
        X = self.conv(X)
        size = list(X.size())
        X = X.view(N, -1, size[1], size[2])
        shape = [0, 2, 1, 1]
        shape[self.axis] = 3
        X = X.permute(shape)
        return X


class MaxPool1(th.nn.Module):

    def __init__(self, axis, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool1, self).__init__()
        if axis not in [2, 3]:
            raise ValueError('Only support 2 or 3 for N-C-H-W-2')
        self.axis = axis
        self.pool = th.nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, X):

        N, C, H, W = X.size()
        axis = 5 - self.axis
        X = X.permute(0, self.axis, 1, axis)
        size = list(X.size())
        X = X.contiguous().view(size[0] * size[1], size[2], size[3])
        # X = X.reshape(size[0] * size[1], size[2], size[3])
        X = self.pool(X)
        size = list(X.size())
        X = X.view(N, -1, size[1], size[2])
        shape = [0, 2, 1, 1]
        shape[self.axis] = 3
        X = X.permute(shape)
        return X


class Conv2(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv2, self).__init__()
        if stride is None:
            stride = kernel_size
        if type(kernel_size) is tuple or list:
            kh, kw = kernel_size
        elif int:
            kh = kw = kernel_size
        if type(stride) is tuple or list:
            sh, sw = stride
        elif int:
            sh = sw = stride
        if type(padding) is tuple or list:
            ph, pw = padding
        elif int:
            ph = pw = padding
        if type(dilation) is tuple or list:
            dh, dw = dilation
        elif int:
            dh = dw = dilation

        self.convh = Conv1(2, in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kh, stride=sh, padding=ph,
                           dilation=dh, groups=groups, bias=bias, padding_mode=padding_mode)
        self.convw = Conv1(3, in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kw, stride=sw, padding=pw,
                           dilation=dw, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, X):
        X = self.convh(X)
        X = self.convw(X)
        return X


class MaxPool2(th.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool2, self).__init__()
        if stride is None:
            stride = kernel_size
        if type(kernel_size) is tuple or list:
            kh, kw = kernel_size
        elif int:
            kh = kw = kernel_size
        if type(stride) is tuple or list:
            sh, sw = stride
        elif int:
            sh = sw = stride
        if type(padding) is tuple or list:
            ph, pw = padding
        elif int:
            ph = pw = padding
        if type(dilation) is tuple or list:
            dh, dw = dilation
        elif int:
            dh = dw = dilation
        self.poolh = MaxPool1(2, kernel_size=kh, stride=sh, padding=ph,
                              dilation=dh, return_indices=return_indices, ceil_mode=ceil_mode)
        self.poolw = MaxPool1(2, kernel_size=kw, stride=sw, padding=pw,
                              dilation=dw, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, X):
        X = self.poolh(X)
        X = self.poolw(X)
        return X


if __name__ == '__main__':
    import numpy as np
    import torch as th
    import torchbox as tb

    shape = 'same'
    ftshift = False
    x_np = np.array([1, 2, 3, 4, 5])
    h_np = np.array([1 + 2j, 2, 3, 4, 5, 6, 7])

    x_th = th.tensor(x_np)
    h_th = th.tensor(h_np)
    x_th = th.stack([x_th, th.zeros(x_th.size())], dim=-1)
    h_th = th.stack([h_th.real, h_th.imag], dim=-1)

    y1 = tb.fftconv1(x_th, h_th, cdim=-1, dim=0, nfft=None, shape=shape, ftshift=ftshift)

    fftconv1layer = FFTConv1([x_th.shape[0]], h=h_th, cdim=-1, dim=0, nfft=None, shape=shape)

    for p in fftconv1layer.parameters():
        print(p)

    y2 = fftconv1layer.forward(x_th)
    # y2 = th.view_as_complex(y2)
    y2 = y2.cpu().detach()

    # print(y1)
    # print(y2)
    print(th.sum(th.abs(y1 - y2)), th.sum(th.angle(y1) - th.angle(y2)))
