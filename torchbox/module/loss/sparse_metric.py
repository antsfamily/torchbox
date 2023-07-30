#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : sparse_metric.py
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


class LogSparseLoss(th.nn.Module):
    """Log sparse loss

    Parameters
    ----------
    X : array
        the input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. The default is :obj:`None`, which means all. 
    lambd : float
        weight, default is 1.
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         loss
    """

    def __init__(self, lambd=1., cdim=None, dim=None, keepdim=False, reduction='mean'):
        super(LogSparseLoss, self).__init__()
        self.lambd = lambd
        self.cdim = cdim
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):  # complex in complex
            X = X.abs()
        else:
            if self.cdim is None:  # real
                X = X.abs()
            else:  # complex in real
                X = th.sum(X**2, dim=self.cdim, keepdims=self.keepdim).sqrt()

        if self.dim is None:
            S = th.sum(th.log2(1 + X / self.lambd))
        else:
            S = th.sum(th.log2(1 + X / self.lambd), self.dim)

        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


class FourierLogSparseLoss(th.nn.Module):
    r"""FourierLogSparseLoss

    Parameters
    ----------
    X : array
        the input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. The default is :obj:`None`, which means all. 
    lambd : float
        weight, default is 1.
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         loss

    """

    def __init__(self, lambd=1., cdim=None, dim=None, keepdim=False, reduction='mean'):
        super(FourierLogSparseLoss, self).__init__()
        self.lambd = lambd
        self.cdim = cdim
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):  # complex in complex
            pass
        else:
            if self.cdim is None:  # real
                pass
            else:  # complex in real
                d = X.ndim
                idxreal = [[0]] if self.keepdim else [0]
                idximag = [[1]] if self.keepdim else [1]
                idxreal = tb.sl(d, axis=self.cdim, idx=idxreal)
                idximag = tb.sl(d, axis=self.cdim, idx=idximag)
                X = X[idxreal] + 1j * X[idximag]

        for a in self.dim:
            X = th.fft.fft(X, n=None, dim=a)

        X = X.abs()

        if self.dim is None:
            S = th.sum(th.log2(1 + X / self.lambd))
        else:
            S = th.sum(th.log2(1 + X / self.lambd), self.dim)

        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


if __name__ == '__main__':

    lambd = 1
    lambd = 2
    lambd = 0.5
    X = th.randn(1, 3, 4, 2)
    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]

    sparse_func = LogSparseLoss(lambd=lambd)
    sparse_func = LogSparseLoss(lambd=lambd, dim=None, cdim=-1)
    sparse_func1 = LogSparseLoss(lambd=lambd, dim=None, cdim=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)

    sparse_func = FourierLogSparseLoss(lambd=lambd)
    sparse_func = FourierLogSparseLoss(lambd=lambd, dim=(1, 2), cdim=-1)
    sparse_func1 = FourierLogSparseLoss(lambd=lambd, dim=(2, 3), keepdim=True, cdim=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)
