#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : gaborconv2d.py
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma    # [channel_out]
    sigma_y = sigma.float() / gamma     # element-wize division, [channel_out]

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax + 1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax + 1)
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    y = y.to(theta.device)
    x = x.to(theta.device)

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
        * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb


class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0):
        super(GaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding

        self.Lambda = nn.Parameter(torch.rand(channel_out), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.psi = nn.Parameter(torch.randn(channel_out) * 0.02, requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(channel_out) * 0.0, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        theta = self.sigmoid(self.theta) * math.pi * 2.0
        gamma = 1.0 + (self.gamma * 0.5)
        sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        Lambda = 0.001 + (self.sigmoid(self.Lambda) * 0.999)
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)
        kernel = kernel.float()   # [channel_out, channel_in, kernel, kernel]

        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out
