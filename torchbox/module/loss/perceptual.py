#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : perceptual.py
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
import torchbox as tb


class RandomProjectionLoss(th.nn.Module):
    r"""RandomProjection loss



    """

    def __init__(self, mode='real', baseloss='MSE', channels=[3, 32], kernel_sizes=[(3, 3)], activations=['ReLU'], reduction='mean'):
        super(RandomProjectionLoss, self).__init__()
        self.mode = mode
        self.reduction = reduction

        nLayers = len(kernel_sizes)
        self.nLayers = nLayers

        net = []
        if mode == 'real':
            conv = th.nn.Conv2d
            bn = th.nn.BatchNorm2d
        if mode == 'complex':
            conv = tb.ComplexConv2d
            bn = tb.ComplexBatchNorm2d

        for n in range(nLayers):
            net.append(conv(channels[n], channels[n + 1], kernel_sizes[n]))
            net.append(bn(channels[n + 1]))
            net.append(eval('th.nn.' + activations[n])())
            self.N = 3

        self.rpf = th.nn.ModuleList(net)
        if baseloss in ['MSE', 'mse']:
            self.baseloss = th.nn.MSELoss(reduction=reduction)
        if baseloss in ['MAE', 'mae']:
            self.baseloss = th.nn.MAELoss(reduction=reduction)

        self.weight_init()

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """   
        with th.no_grad():
            loss = 0.
            for n in range(self.nLayers * self.N):
                P, G = self.rpf[n](P), self.rpf[n](G)
                if n % self.N == self.N - 1:
                    loss += self.baseloss(P, G)

            if self.reduction == 'mean':
                loss /= self.nLayers
            if self.reduction == 'sum':
                pass
        return loss

    def weight_init(self):

        for n in range(self.nLayers):
            if self.mode in ['Complex', 'complex']:
                th.nn.init.orthogonal_(self.rpf[n * self.N].convr.weight, th.nn.init.calculate_gain('leaky_relu'))
                th.nn.init.orthogonal_(self.rpf[n * self.N].convi.weight, th.nn.init.calculate_gain('leaky_relu'))
            else:
                th.nn.init.orthogonal_(self.rpf[n * self.N].weight, th.nn.init.calculate_gain('leaky_relu'))

        for param in self.parameters():
            param.requires_grad = False


if __name__ == '__main__':

    loss_func = RandomProjectionLoss(mode='real', baseloss='MSE', channels=[3, 4], kernel_sizes=[(3, 3)], activations=['LeakyReLU'], reduction='mean')
    P = th.randn(4, 3, 64, 64)
    G = th.randn(4, 3, 64, 64)
    S = loss_func(P, G)
    print(S)

    loss_func = RandomProjectionLoss(mode='complex', baseloss='MSE', channels=[3, 4], kernel_sizes=[(3, 3)], activations=['LeakyReLU'], reduction='mean')
    P = th.randn(4, 3, 64, 64, 2)
    G = th.randn(4, 3, 64, 64, 2)
    S = loss_func(P, G)
    print(S)
