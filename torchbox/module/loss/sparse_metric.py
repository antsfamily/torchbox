#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : sparse_metric.py
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
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. The default is :obj:`None`, which means all. 
    lambd : float
        weight, default is 1.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         loss
    """

    def __init__(self, lambd=1., cdim=None, dim=None, keepcdim=False, reduction='mean'):
        super(LogSparseLoss, self).__init__()
        self.lambd = lambd
        self.cdim = cdim
        self.dim = dim
        self.keepcdim = keepcdim
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):  # complex in complex
            X = X.abs()
        else:
            if self.cdim is None:  # real
                X = X.abs()
            else:  # complex in real
                X = th.sum(X**2, dim=self.cdim, keepdims=self.keepcdim).sqrt()

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
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. The default is :obj:`None`, which means all. 
    lambd : float
        weight, default is 1.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         loss

    """

    def __init__(self, lambd=1., cdim=None, dim=None, keepcdim=False, reduction='mean'):
        super(FourierLogSparseLoss, self).__init__()
        self.lambd = lambd
        self.cdim = cdim
        self.dim = dim
        self.keepcdim = keepcdim
        self.reduction = reduction

    def forward(self, X):
        if th.is_complex(X):  # complex in complex
            pass
        else:
            if self.cdim is None:  # real
                pass
            else:  # complex in real
                d = X.ndim
                idxreal = [[0]] if self.keepcdim else [0]
                idximag = [[1]] if self.keepcdim else [1]
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
    sparse_func1 = FourierLogSparseLoss(lambd=lambd, dim=(2, 3), keepcdim=True, cdim=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)
