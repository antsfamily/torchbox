#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : contrast.py
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


def contrast(X, mode='way1', cdim=None, dim=None, keepdim=False, reduction=None):
    r"""Compute contrast of an complex image

    ``'way1'`` is defined as follows, see [1]:

    .. math::
       C = \frac{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}{{\rm E}(|I|^2)}


    ``'way2'`` is defined as follows, see [2]:

    .. math::
        C = \frac{{\rm E}(|I|^2)}{\left({\rm E}(|I|)\right)^2}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"

    Parameters
    ----------
    X : torch tensor
        The image array.
    mode : str, optional
        ``'way1'`` or ``'way2'``
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : tuple, None, optional
        The dimension axis for computing contrast. 
        The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    C : scalar or tensor
        The contrast value of input.
    
    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)

        # real
        C1 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction=None)
        C2 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='sum')
        C3 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction=None)
        C2 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction='sum')
        C3 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        C1 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction=None)
        C2 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='sum')
        C3 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='mean')
        print(C1, C2, C3)


        # output
        tensor([[1.2612, 1.1085],
                [1.5992, 1.2124],
                [0.8201, 0.9887],
                [1.4376, 1.0091],
                [1.1397, 1.1860]]) tensor(11.7626) tensor(1.1763)
        tensor([0.6321, 1.1808, 0.5884, 1.1346, 0.6038]) tensor(4.1396) tensor(0.8279)
        tensor([0.6321, 1.1808, 0.5884, 1.1346, 0.6038]) tensor(4.1396) tensor(0.8279)
    """

    X = tb.pow(X, cdim=cdim, keepdim=True)

    if mode in ['way1', 'WAY1']:
        Xmean = X.mean(dim=dim, keepdims=True)
        C = (X - Xmean).pow(2).mean(dim=dim, keepdims=True).sqrt() / (Xmean + tb.EPS)
        C = th.sum(C, dim=dim, keepdims=True)
    if mode in ['way2', 'WAY2']:
        C = X.mean(dim=dim, keepdims=True) / ((X.sqrt().mean(dim=dim, keepdims=True)).pow(2) + tb.EPS)
        C = th.sum(C, dim=dim, keepdims=True)

    sdim = tb.dimreduce(C.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(C, dim=sdim, keepdim=keepdim, reduction=reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)

    # real
    C1 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction=None)
    C2 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='sum')
    C3 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='mean')
    print(C1, C2, C3)

    # complex in real format
    C1 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction=None)
    C2 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction='sum')
    C3 = contrast(X, cdim=1, dim=(-2, -1), mode='way1', reduction='mean')
    print(C1, C2, C3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    C1 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction=None)
    C2 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='sum')
    C3 = contrast(X, cdim=None, dim=(-2, -1), mode='way1', reduction='mean')
    print(C1, C2, C3)

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)
    # real
    C1 = contrast(X, cdim=None, dim=None, mode='way1', keepdim=True, reduction=None)
    C2 = contrast(X, cdim=None, dim=None, mode='way1', keepdim=True, reduction='sum')
    C3 = contrast(X, cdim=None, dim=None, mode='way1', keepdim=True, reduction='mean')
    print(C1, C2, C3)

    # complex in real format
    C1 = contrast(X, cdim=1, dim=None, mode='way1', keepdim=True, reduction=None)
    C2 = contrast(X, cdim=1, dim=None, mode='way1', keepdim=True, reduction='sum')
    C3 = contrast(X, cdim=1, dim=None, mode='way1', keepdim=True, reduction='mean')
    print(C1, C2, C3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    C1 = contrast(X, cdim=None, dim=None, mode='way1', keepdim=True, reduction=None)
    C2 = contrast(X, cdim=None, dim=None, mode='way1', keepdim=True, reduction='sum')
    C3 = contrast(X, cdim=None, dim=None, mode='way1', keepdim=True, reduction='mean')
    print(C1, C2, C3)
