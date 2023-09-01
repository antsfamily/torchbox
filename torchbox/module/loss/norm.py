#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : norm.py
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


class FnormLoss(th.nn.Module):
    r"""F-norm Loss

    Both complex and real representation are supported.

    .. math::
       {\rm norm}({\bf P}) = \|{\bf P}\|_2 = \left(\sum_{x_i\in {\bf P}}|x_i|^2\right)^{\frac{1}{2}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, None or optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    tensor
         the inputs's f-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        print('---norm')

        # real
        F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        # complex in real format
        F1 = FnormLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        F2 = FnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        F3 = FnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        ---norm
        tensor([[3.0401, 4.9766],
                [4.8830, 3.1261],
                [6.3124, 4.1407],
                [5.9283, 4.5896],
                [3.4909, 6.7252]]) tensor(47.2130) tensor(4.7213)
        tensor([5.8317, 5.7980, 7.5493, 7.4973, 7.5772]) tensor(34.2535) tensor(6.8507)
        tensor([5.8317, 5.7980, 7.5493, 7.4973, 7.5772]) tensor(34.2535) tensor(6.8507)
    """

    def __init__(self, cdim=None, dim=None, keepdim=False, reduction='mean'):
        super(FnormLoss, self).__init__()
        self.cdim = cdim
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = reduction

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """   
        return tb.norm(P - G, mode='fro', cdim=self.cdim, dim=self.dim, keepdim=self.keepdim, reduction=self.reduction)


class PnormLoss(th.nn.Module):
    r"""obtain the p-norm of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm pnorm}({\bf P}) = \|{\bf P}\|_p = \left(\sum_{x_i\in {\bf P}}|x_i|^p\right)^{\frac{1}{p}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    p : int
        Specifies the power. The default is 2.
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, None or optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')
    
    Returns
    -------
    tensor
         the inputs's p-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        print('---norm')

        # real
        F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        # complex in real format
        F1 = PnormLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        F2 = PnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        F3 = PnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        ---norm
        tensor([[3.0401, 4.9766],
                [4.8830, 3.1261],
                [6.3124, 4.1407],
                [5.9283, 4.5896],
                [3.4909, 6.7252]]) tensor(47.2130) tensor(4.7213)
        tensor([5.8317, 5.7980, 7.5493, 7.4973, 7.5772]) tensor(34.2535) tensor(6.8507)
        tensor([5.8317, 5.7980, 7.5493, 7.4973, 7.5772]) tensor(34.2535) tensor(6.8507)
    """

    def __init__(self, p=2, cdim=None, dim=None, keepdim=False, reduction='mean'):
        super(PnormLoss, self).__init__()
        self.p = p
        self.dim = dim
        self.cdim = cdim
        self.keepdim = keepdim
        self.reduction = reduction

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """   
        return tb.norm(P - G, mode='pnorm%d'%self.p, cdim=self.cdim, dim=self.dim, keepdim=self.keepdim, reduction=self.reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)
    print('---fnorm')

    # real
    F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
    F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
    F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
    print(F1, F2, F3)

    # complex in real format
    F1 = FnormLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
    F2 = FnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
    F3 = FnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
    print(F1, F2, F3)

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
    F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
    F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
    print(F1, F2, F3)

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)
    print('---pnorm')
    
    # real
    F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
    F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
    F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
    print(F1, F2, F3)

    # complex in real format
    F1 = PnormLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
    F2 = PnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
    F3 = PnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
    print(F1, F2, F3)

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
    F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
    F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
    print(F1, F2, F3)
