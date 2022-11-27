#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : contrast.py
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


class Contrast(th.nn.Module):
    r"""Contrast

    way1 is defined as follows, see [1]:

    .. math::
       C = \frac{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}{{\rm E}(|I|^2)}


    way2 is defined as follows, see [2]:

    .. math::
        C = \frac{{\rm E}(|I|^2)}{\left({\rm E}(|I|)\right)^2}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"

    Parameters
    ----------
    X : torch tensor
        The image tensor.
    mode : str, optional
        ``'way1'`` or ``'way2'``
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing contrast. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

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
        C1 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
        C2 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
        C3 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        # complex in real format
        C1 = Contrast(mode='way1', cdim=1, dim=(-2, -1), reduction=None)(X)
        C2 = Contrast(mode='way1', cdim=1, dim=(-2, -1), reduction='sum')(X)
        C3 = Contrast(mode='way1', cdim=1, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        C1 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
        C2 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
        C3 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
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

    def __init__(self, mode='way1', cdim=None, dim=None, keepcdim=False, reduction='mean'):
        super(Contrast, self).__init__()
        self.mode = mode
        self.cdim = cdim
        self.dim = dim
        self.keepcdim = keepcdim
        self.reduction = reduction

    def forward(self, X):

        return tb.contrast(X, mode=self.mode, cdim=self.cdim, dim=self.dim, keepcdim=self.keepcdim, reduction=self.reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)

    # real
    C1 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
    C2 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
    C3 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
    print(C1, C2, C3)

    # complex in real format
    C1 = Contrast(mode='way1', cdim=1, dim=(-2, -1), reduction=None)(X)
    C2 = Contrast(mode='way1', cdim=1, dim=(-2, -1), reduction='sum')(X)
    C3 = Contrast(mode='way1', cdim=1, dim=(-2, -1), reduction='mean')(X)
    print(C1, C2, C3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    C1 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
    C2 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
    C3 = Contrast(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
    print(C1, C2, C3)