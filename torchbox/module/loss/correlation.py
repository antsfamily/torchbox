#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : entropy.py
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


class CosSimLoss(th.nn.Module):
    r"""compute the cosine similarity loss of the inputs

    If utilize the amplitude of pearson correlation as loss

    .. math::
       {\mathcal L} = 1 - |\frac{<{\bf p}, {\bf g}>}{\|p\|_2\|g\|_2}|

    If utilize the angle of pearson correlation as loss
    
    .. math::
       {\mathcal L} = |\angle \frac{<{\bf p}, {\bf g}>}{\|p\|_2\|g\|_2}|

    Parameters
    ----------
    P : tensor
        the first/left input, such as the predicted
    G : tensor
        the second/right input, such as the ground-truth
    mode : str
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity.
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis for computing entropy. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : tensor
        The entropy of the inputs.
    
    Examples
    --------

    ::

        import torch as th
        from torchbox import PeaCorLoss

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        S1 = CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        S2 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        S3 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in real format
        S1 = CosSimLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        S2 = CosSimLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        S3 = CosSimLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        S1 = CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        S2 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        S3 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # output
        tensor([[0.4791, 0.0849],
                [0.0334, 0.4855],
                [0.0136, 0.2280],
                [0.4951, 0.2166],
                [0.4484, 0.4221]]) tensor(2.9068) tensor(0.2907)
        tensor([[0.2926],
                [0.2912],
                [0.1505],
                [0.3993],
                [0.3350]]) tensor([1.4685]) tensor([0.2937])
        tensor([0.2926, 0.2912, 0.1505, 0.3993, 0.3350]) tensor(1.4685) tensor(0.2937)
    
    """

    def __init__(self, mode='abs', cdim=None, dim=None, keepdim=False, reduction='mean'):
        super(CosSimLoss, self).__init__()
        self.mode = mode
        self.cdim = cdim
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = reduction

    def forward(self, P, G):
        
        S = tb.cossim(P, G, mode=self.mode, cdim=self.cdim, dim=self.dim, keepdim=True, reduction=None)

        if self.mode in ['abs', 'amplitude']:
            S = 1 - S
        elif self.mode in ['angle', 'phase']:
            S = S.abs()

        sdim = tb.rdcdim(S.ndim, cdim=self.cdim, dim=self.dim, keepcdim=False, reduction=self.reduction)

        return tb.reduce(S, dim=sdim, keepdim=self.keepdim, reduction=self.reduction)

class PeaCorLoss(th.nn.Module):
    r"""compute the pearson correlation loss of the inputs

    If utilize the amplitude of pearson correlation as loss

    .. math::
       {\mathcal L} = 1 - |\frac{<{\bf p}, {\bf g}>}{\|p\|_2\|g\|_2}|

    If utilize the angle of pearson correlation as loss

    .. math::
       {\mathcal L} = |\angle \frac{<{\bf p}, {\bf g}>}{\|p\|_2\|g\|_2}|
       
    where :math:`p` and :math:`g` is the centered version (removed mean) of inputs

    Parameters
    ----------
    P : tensor
        the first/left input, such as the predicted
    G : tensor
        the second/right input, such as the ground-truth
    mode : str
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity.
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis for computing entropy. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : tensor
        The entropy of the inputs.
    
    Examples
    --------

    ::

        import torch as th
        from torchbox import PeaCorLoss

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        S1 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        S2 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        S3 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in real format
        S1 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        S2 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        S3 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        S1 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        S2 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        S3 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)


        # output
        tensor([[0.6010, 0.0260],
                [0.0293, 0.4981],
                [0.0063, 0.2284],
                [0.3203, 0.2851],
                [0.3757, 0.3936]]) tensor(2.7639) tensor(0.2764)
        tensor([[0.3723],
                [0.2992],
                [0.1267],
                [0.3020],
                [0.2910]]) tensor([1.3911]) tensor([0.2782])
        tensor([0.3723, 0.2992, 0.1267, 0.3020, 0.2910]) tensor(1.3911) tensor(0.2782)
    
    """

    def __init__(self, mode='abs', cdim=None, dim=None, keepdim=False, reduction='mean'):
        super(PeaCorLoss, self).__init__()
        self.mode = mode.lower()
        self.cdim = cdim
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = reduction

    def forward(self, P, G):

        S = tb.peacor(P, G, mode=self.mode, cdim=self.cdim, dim=self.dim, keepdim=True, reduction=None)

        if self.mode in ['abs', 'amplitude']:
            S = 1 - S
        elif self.mode in ['angle', 'phase']:
            S = S.abs()

        sdim = tb.rdcdim(S.ndim, cdim=self.cdim, dim=self.dim, keepcdim=False, reduction=self.reduction)

        return tb.reduce(S, dim=sdim, keepdim=self.keepdim, reduction=self.reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    S1 = CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
    S2 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
    S3 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
    print(S1, S2, S3)

    # complex in real format
    S1 = CosSimLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
    S2 = CosSimLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
    S3 = CosSimLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
    print(S1, S2, S3)

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    S1 = CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
    S2 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
    S3 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
    print(S1, S2, S3)

    print('---PeaCorLoss')

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    S1 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
    S2 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
    S3 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
    print(S1, S2, S3)

    # complex in real format
    S1 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
    S2 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
    S3 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
    print(S1, S2, S3)

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    S1 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
    S2 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
    S3 = PeaCorLoss(cdim=None, mode='abs', dim=(-2, -1), reduction='mean')(P, G)
    print(S1, S2, S3)
