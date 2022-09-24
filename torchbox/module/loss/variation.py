#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

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
            X = (X.real**2 + X.imag**2).sqrt()
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
