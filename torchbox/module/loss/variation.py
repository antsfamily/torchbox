#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : variation.py
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
from torchbox.utils.const import EPS
from torchbox.base.arrayops import sl


class TotalVariation(th.nn.Module):
    r"""Total Variarion



    """

    def __init__(self, reduction='mean', axis=0):
        super(TotalVariation, self).__init__()
        self.reduction = reduction
        if type(axis) is int:
            self.axis = [axis]
        else:
            self.axis = list(axis)

    def forward(self, X):

        if th.is_complex(X):
            X = (X.real*X.real + X.imag*X.imag).sqrt()
        elif X.size(-1) == 2:
            X = X.pow(2).sum(axis=-1).sqrt()

        D = X.dim()
        # compute gradients in axis direction
        for a in self.axis:
            d = X.size(a)
            X = (X[sl(D, a, range(1, d))] - X[sl(D, a, range(0, d - 1))]).abs()

        G = th.mean(X, self.axis, keepdim=True)

        if self.reduction == 'mean':
            V = th.mean(G)
        if self.reduction == 'sum':
            V = th.sum(G)

        return -th.log(V)


if __name__ == '__main__':

    tv_func = TotalVariation(reduction='mean', axis=1)
    X = th.randn(1, 3, 4, 2)
    V = tv_func(X)
    print(V)

    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
    V = tv_func(X)
    print(V)
