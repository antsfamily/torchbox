#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : variation.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 0.0
# @license   : The Apache License 2.0
# @note      : 
# 
# The Apache 2.0 License
# Copyright (C) 2013- Zhi Liu
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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
