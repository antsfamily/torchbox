#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : phase_convolution.py
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
import torch.nn.functional as F


class PhaseConv1d(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=None, padding_mode='zeros'):
        super(PhaseConv1d, self).__init__()
        self.weight = th.nn.Parameter(
            th.zeros(out_channels, int(in_channels / groups), kernel_size))
        if (bias is None) or (not bias):
            self.bias = None
        else:
            self.bias = th.nn.Parameter(th.zeros(out_channels, 2))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        weight = th.exp(1j * self.weight)
        weight = th.fft.ifft(weight, dim=-1)
        weight = th.view_as_real(weight)
        if self.bias is None:
            x = th.stack((F.conv1d(x[..., 0], weight[..., 0], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups),
                          F.conv1d(x[..., 1], weight[..., 1], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)), dim=-1)
        else:
            x = th.stack((F.conv1d(x[..., 0], weight[..., 0], bias=self.bias[..., 0], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups),
                          F.conv1d(x[..., 1], weight[..., 1], bias=self.bias[..., 1], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)), dim=-1)
        return x


class PhaseConv2d(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=None, padding_mode='zeros'):
        super(PhaseConv2d, self).__init__()
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * 2

        self.weight = th.nn.Parameter(th.zeros(out_channels, int(
            in_channels / groups), kernel_size[0], kernel_size[1]))
        if (bias is None) or (not bias):
            self.bias = None
        else:
            self.bias = th.nn.Parameter(th.zeros(out_channels, 2))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        weight = th.exp(1j * self.weight)
        weight = th.fft.ifft(weight, dim=-1)
        weight = th.fft.ifft(weight, dim=-2)

        weight = th.view_as_real(weight)

        if self.bias is None:
            x = th.stack((F.conv2d(x[..., 0], weight[..., 0], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups),
                          F.conv2d(x[..., 1], weight[..., 1], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)), dim=-1)
        else:
            x = th.stack((F.conv2d(x[..., 0], weight[..., 0], bias=self.bias[..., 0], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups),
                          F.conv2d(x[..., 1], weight[..., 1], bias=self.bias[..., 1], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)), dim=-1)
        return x


class ComplexPhaseConv1d(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=None, padding_mode='zeros'):
        super(ComplexPhaseConv1d, self).__init__()
        self.weight = th.nn.Parameter(
            th.zeros(out_channels, int(in_channels / groups), kernel_size))
        if (bias is None) or (not bias):
            self.bias = None
        else:
            self.bias = th.nn.Parameter(th.zeros(out_channels, 2))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        weight = th.exp(1j * self.weight)
        weight = th.fft.ifft(weight, dim=-1)
        weight = th.view_as_real(weight)
        if self.bias is None:
            x = th.stack((F.conv1d(x[..., 0], weight[..., 0], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) -
                          F.conv1d(x[..., 1], weight[..., 1], bias=self.bias, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups),
                          F.conv1d(x[..., 1], weight[..., 0], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) +
                          F.conv1d(x[..., 0], weight[..., 1], bias=self.bias, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups)
                          ), dim=-1)
        else:
            x = th.stack((F.conv1d(x[..., 0], weight[..., 0], bias=self.bias[..., 0], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) -
                          F.conv1d(x[..., 1], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups),
                          F.conv1d(x[..., 1], weight[..., 0], bias=self.bias[..., 0], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) +
                          F.conv1d(x[..., 0], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups)
                          ), dim=-1)
        return x


class ComplexPhaseConv2d(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=None, padding_mode='zeros'):
        super(ComplexPhaseConv2d, self).__init__()
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * 2

        self.weight = th.nn.Parameter(th.zeros(out_channels, int(in_channels / groups), kernel_size[0], kernel_size[1]))
        if (bias is None) or (not bias):
            self.bias = None
        else:
            self.bias = th.nn.Parameter(th.zeros(out_channels, 2))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        weight = th.exp(1j * self.weight)
        weight = th.fft.ifft(weight, dim=-1)
        weight = th.fft.ifft(weight, dim=-2)

        weight = th.view_as_real(weight)

        if self.bias is None:
            x = th.stack((F.conv2d(x[..., 0], weight[..., 0], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) -
                          F.conv2d(x[..., 1], weight[..., 1], bias=self.bias, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups),
                          F.conv2d(x[..., 1], weight[..., 0], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) +
                          F.conv2d(x[..., 0], weight[..., 1], bias=self.bias, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups)
                          ), dim=-1)
        else:
            x = th.stack((F.conv2d(x[..., 0], weight[..., 0], bias=self.bias[..., 0], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) -
                          F.conv2d(x[..., 1], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups),
                          F.conv2d(x[..., 1], weight[..., 0], bias=self.bias[..., 0], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) +
                          F.conv2d(x[..., 0], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups)
                          ), dim=-1)
        return x


class PhaseConvTranspose1d(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=None, dilation=1, padding_mode='zeros'):
        super(PhaseConvTranspose1d, self).__init__()
        self.weight = th.nn.Parameter(th.zeros(in_channels, int(out_channels / groups), kernel_size))
        if (bias is None) or (not bias):
            self.bias = None
        else:
            self.bias = th.nn.Parameter(th.zeros(out_channels, 2))

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        weight = th.exp(1j * self.weight)
        weight = th.fft.ifft(weight, dim=-1)
        weight = th.view_as_real(weight)

        if self.bias is None:
            x = th.stack((F.conv_transpose1d(x[..., 0], weight[..., 0], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation),
                          F.conv_transpose1d(x[..., 1], weight[..., 1], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)), dim=-1)
        else:
            x = th.stack((F.conv_transpose1d(x[..., 0], weight[..., 0], bias=self.bias[..., 0], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation),
                          F.conv_transpose1d(x[..., 1], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)), dim=-1)
        return x


class PhaseConvTranspose2d(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=None, dilation=1, padding_mode='zeros'):
        super(PhaseConvTranspose2d, self).__init__()
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * 2

        self.weight = th.nn.Parameter(th.zeros(in_channels, int(out_channels / groups), kernel_size[0], kernel_size[1]))
        if (bias is None) or (not bias):
            self.bias = None
        else:
            self.bias = th.nn.Parameter(th.zeros(out_channels, 2))

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        weight = th.exp(1j * self.weight)
        weight = th.fft.ifft(weight, dim=-1)
        weight = th.fft.ifft(weight, dim=-2)
        weight = th.view_as_real(weight)

        if self.bias is None:
            x = th.stack((F.conv_transpose2d(x[..., 0], weight[..., 0], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation),
                          F.conv_transpose2d(x[..., 1], weight[..., 1], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)), dim=-1)
        else:
            x = th.stack((F.conv_transpose2d(x[..., 0], weight[..., 0], bias=self.bias[..., 0], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation),
                          F.conv_transpose2d(x[..., 1], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)), dim=-1)
        return x


class ComplexPhaseConvTranspose1d(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=None, dilation=1, padding_mode='zeros'):
        super(ComplexPhaseConvTranspose1d, self).__init__()
        self.weight = th.nn.Parameter(th.zeros(in_channels, int(out_channels / groups), kernel_size))
        if (bias is None) or (not bias):
            self.bias = None
        else:
            self.bias = th.nn.Parameter(th.zeros(out_channels, 2))

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        weight = th.exp(1j * self.weight)
        weight = th.fft.ifft(weight, dim=-1)
        weight = th.view_as_real(weight)

        if self.bias is None:
            x = th.stack((F.conv_transpose1d(x[..., 0], weight[..., 0], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation) -
                          F.conv_transpose1d(x[..., 1], weight[..., 1], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation),
                          F.conv_transpose1d(x[..., 1], weight[..., 0], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation) +
                          F.conv_transpose1d(x[..., 0], weight[..., 1], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)), dim=-1)
        else:
            x = th.stack((F.conv_transpose1d(x[..., 0], weight[..., 0], bias=self.bias[..., 0], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation) -
                          F.conv_transpose1d(x[..., 1], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation),
                          F.conv_transpose1d(x[..., 1], weight[..., 0], bias=self.bias[..., 0], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation) +
                          F.conv_transpose1d(x[..., 0], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)), dim=-1)
        return x


class ComplexPhaseConvTranspose2d(th.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=None, dilation=1, padding_mode='zeros'):
        super(ComplexPhaseConvTranspose2d, self).__init__()
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * 2

        self.weight = th.nn.Parameter(th.zeros(in_channels, int(out_channels / groups), kernel_size[0], kernel_size[1]))
        if (bias is None) or (not bias):
            self.bias = None
        else:
            self.bias = th.nn.Parameter(th.zeros(out_channels, 2))

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        weight = th.exp(1j * self.weight)
        weight = th.fft.ifft(weight, dim=-1)
        weight = th.fft.ifft(weight, dim=-2)
        weight = th.view_as_real(weight)

        if self.bias is None:
            x = th.stack((F.conv_transpose2d(x[..., 0], weight[..., 0], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation) -
                          F.conv_transpose2d(x[..., 1], weight[..., 1], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation),
                          F.conv_transpose2d(x[..., 1], weight[..., 0], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation) +
                          F.conv_transpose2d(x[..., 0], weight[..., 1], bias=self.bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)), dim=-1)
        else:
            x = th.stack((F.conv_transpose2d(x[..., 0], weight[..., 0], bias=self.bias[..., 0], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation) -
                          F.conv_transpose2d(x[..., 1], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation),
                          F.conv_transpose2d(x[..., 1], weight[..., 0], bias=self.bias[..., 0], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation) +
                          F.conv_transpose2d(x[..., 0], weight[..., 1], bias=self.bias[..., 1], stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)), dim=-1)
        return x


if __name__ == '__main__':

    import torch as th

    device = th.device('cuda:0')

    N, L = 6, 4

    x = th.randn(N, 1, L, 2)
    t = th.randn(N, 3, L, 2)

    pconv = PhaseConv1d(1, 3, 3, 1, 1, bias=True)

    pconv = pconv.to(device)
    x, t = x.to(device), t.to(device)

    y = pconv(x)

    loss_fn = th.nn.MSELoss()

    loss = loss_fn(y, t)

    loss.backward()

    print(x.shape)
    print(y.shape)
    print(loss.item())

    N, H, W = 6, 16, 8

    x = th.randn(N, 1, H, W, 2)
    t = th.randn(N, 5, H, W, 2)

    pconv = PhaseConv2d(1, 5, 3, 2, 1, bias=True)
    pconvt = PhaseConvTranspose2d(5, 1, 3, 2, 1, 1, bias=True)

    pconv = pconv.to(device)
    pconvt = pconvt.to(device)
    x, t = x.to(device), t.to(device)

    y = pconv(x)

    loss_fn = th.nn.MSELoss()

    loss = loss_fn(y, t)

    loss.backward()

    print(x.shape)
    print(y.shape)
    print(loss.item())

    z = pconvt(y)
    print("z.shape", z.shape)

