#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchbox as tb


class LogSparseLoss(th.nn.Module):
    """Log sparse loss

    Parameters
    ----------
    X : array
        original
    X : array
        reconstructed
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. The default is :obj:`None`, which means all. 
    p : float
        weight
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         loss
    """

    def __init__(self, λ=1., cdim=None, dim=None, keepcdim=False, reduction='mean'):
        super(LogSparseLoss, self).__init__()
        self.λ = λ
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
            S = th.sum(th.log2(1 + X / self.λ))
        else:
            S = th.sum(th.log2(1 + X / self.λ), self.dim)

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
        original
    X : array
        reconstructed
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. The default is :obj:`None`, which means all. 
    p : float
        weight
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         loss

    """

    def __init__(self, λ=1., cdim=None, dim=None, keepcdim=False, reduction='mean'):
        super(FourierLogSparseLoss, self).__init__()
        self.λ = λ
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
            S = th.sum(th.log2(1 + X / self.λ))
        else:
            S = th.sum(th.log2(1 + X / self.λ), self.dim)

        if self.reduction == 'mean':
            S = th.mean(S)
        if self.reduction == 'sum':
            S = th.sum(S)

        return S


if __name__ == '__main__':

    λ = 1
    λ = 2
    λ = 0.5
    X = th.randn(1, 3, 4, 2)
    X = X[:, :, :, 0] + 1j * X[:, :, :, 1]

    sparse_func = LogSparseLoss(λ=λ)
    sparse_func = LogSparseLoss(λ=λ, dim=None, cdim=-1)
    sparse_func1 = LogSparseLoss(λ=λ, dim=None, cdim=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)

    sparse_func = FourierLogSparseLoss(λ=λ)
    sparse_func = FourierLogSparseLoss(λ=λ, dim=(1, 2), cdim=-1)
    sparse_func1 = FourierLogSparseLoss(λ=λ, dim=(2, 3), keepcdim=True, cdim=1)
    S = sparse_func(X)
    print(S)

    Y = th.view_as_real(X)
    S = sparse_func(Y)
    print(S)

    Y = Y.permute(0, 3, 1, 2)
    S = sparse_func1(Y)
    print(S)

    # print(X)
