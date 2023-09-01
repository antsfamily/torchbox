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


def cossim(P, G, mode=None, cdim=None, dim=None, keepdim=False, reduction=None):
    r"""compute cosine similarity

    .. math::
       s = \frac{<{\bf p}, {\bf g}>}{\|{\bf p}\|_2\|{\bf g}\|_2}

    .. note:: 
       For complex, the magnitude still gives the "similarity" between them, 
       where the complex angle gives the complex phase factor required to fully reach that similarity. 
       refers `Cosine similarity between complex vectors <https://math.stackexchange.com/questions/273527/cosine-similarity-between-complex-vectors#:~:text=For%20complex%2C%20the%20magnitude%20still%20gives%20the%20%22similarity%22,is%20a%20real%20scalar%20multiple%20of%20y%20y.>`_

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
    dim : tuple, None, optional
        The dimension indexes for computing cosine similarity. The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    S : scalar or tensor
        The cosine similarity value of inputs.
    
    see also :func:`~torchbox.evaluation.correlation.peacor`, :func:`~torchbox.evaluation.correlation.eigveccor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.CosSimLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.

    Examples
    --------

    ::

        import torch as th
        from torchbox import cossim

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)
        dim = (-2, -1)

        # real
        C1 = cossim(P, G, cdim=None, dim=dim, reduction=None)
        C2 = cossim(P, G, cdim=None, dim=dim, reduction='sum')
        C3 = cossim(P, G, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = cossim(P, G, cdim=1, dim=dim, reduction=None)
        C2 = cossim(P, G, cdim=1, dim=dim, reduction='sum')
        C3 = cossim(P, G, cdim=1, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = cossim(P, G, cdim=None, dim=dim, reduction=None)
        C2 = cossim(P, G, cdim=None, dim=dim, reduction='sum')
        C3 = cossim(P, G, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

    """

    S = tb.dot(G, P, mode='xyh', cdim=cdim, dim=dim, keepdim=True) / (tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True) * tb.norm(P, mode='fro', cdim=cdim, dim=dim, keepdim=True) + tb.EPS)
    
    if mode is not None:
        if mode.lower() in ['abs', 'amplitude', 'amp']:
            S = tb.abs(S, cdim=cdim, keepdim=True)
        elif mode.lower() in ['angle', 'amplitude', 'amp']:
            S = tb.angle(S, cdim=cdim, keepdim=True)
        else:
            raise ValueError('mode %s is not support!' % mode)

    sdim = tb.dimreduce(S.ndim, cdim=cdim, dim=dim, keepcdim=keepdim, reduction=reduction)

    return tb.reduce(S, dim=sdim, keepdim=keepdim, reduction=reduction)


def peacor(P, G, mode=None, cdim=None, dim=None, keepdim=False, reduction=None):
    r"""compute the Pearson Correlation Coefficient

    The Pearson correlation coefficient can be viewed as the cosine similarity of centered (remove mean) input.

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
    dim : tuple, None, optional
        The dimension axis for computing the Pearson correlation coefficient. 
        The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    S : scalar or tensor
        The Pearson correlation coefficient value of inputs.

    see also :func:`~torchbox.evaluation.correlation.cossim`, :func:`~torchbox.evaluation.correlation.eigveccor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.CosSimLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.

    Examples
    --------

    ::

        import torch as th
        from torchbox import peacor

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)
        dim = (-2, -1)

        # real
        C1 = peacor(P, G, cdim=None, dim=dim, reduction=None)
        C2 = peacor(P, G, cdim=None, dim=dim, reduction='sum')
        C3 = peacor(P, G, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = peacor(P, G, cdim=1, dim=dim, reduction=None)
        C2 = peacor(P, G, cdim=1, dim=dim, reduction='sum')
        C3 = peacor(P, G, cdim=1, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = peacor(P, G, cdim=None, dim=dim, reduction=None)
        C2 = peacor(P, G, cdim=None, dim=dim, reduction='sum')
        C3 = peacor(P, G, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        x = th.randn(2, 3) + 1j*th.randn(2, 3)
        print(th.corrcoef(x))
        print(peacor(x[0], x[1]))
        print(peacor(x[1], x[0]))

    """

    P = P - tb.mean(P, cdim=cdim, dim=dim, keepdim=True)
    G = G - tb.mean(G, cdim=cdim, dim=dim, keepdim=True)

    return cossim(G, P, mode=mode, cdim=cdim, dim=dim, keepdim=keepdim, reduction=reduction)


def eigveccor(P, G, npcs=4, mode=None, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction=None):
    r"""computes cosine similarity of eigenvectors

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    npcs : int, optional
        the number principal components for comparing, by default 4
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`P` and :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` and :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`P` and :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` and :attr:`G` will be treated as real-valued
    sdim : int, optional
        the dimension index of sample, by default -1
    fdim : int, optional
        the dimension index of feature, by default -2
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    S : scalar or tensor
        The eigenvector correlation value of inputs.
    
    see also :func:`~torchbox.evaluation.correlation.cossim`, :func:`~torchbox.evaluation.correlation.peacor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.CosSimLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.

    Examples
    --------

    ::

        import torch as th
        from torchbox import eigveccor
    
        print('---compare eigen vector correlation (complex in real)')
        G = th.randn(2, 3, 2, 64, 4)
        P = th.randn(2, 3, 2, 64, 4)
        print(eigveccor(G, G, npcs=4, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))
        print(eigveccor(G, G, npcs=4, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction=None).shape)
        
        print('---compare eigen vector correlation (complex in complex)')
        G = th.randn(2, 3, 64, 4) + 1j*th.randn(2, 3, 64, 4)
        P = th.randn(2, 3, 64, 4) + 1j*th.randn(2, 3, 64, 4)
        print(eigveccor(G, G, npcs=4, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))
        print(eigveccor(G, G, npcs=4, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction=None).shape)
        
    """

    if cdim is not None:
        sdim = tb.rmcdim(G.ndim, cdim=cdim, dim=sdim, keepdim=keepdim)
        fdim = tb.rmcdim(G.ndim, cdim=cdim, dim=fdim, keepdim=keepdim)
        G = tb.r2c(G, cdim=cdim, keepdim=keepdim)
        P = tb.r2c(P, cdim=cdim, keepdim=keepdim)

    Ug, _ = tb.pcat(G, sdim=sdim, fdim=fdim, isnorm=True, eigbkd='svd')
    Up, _ = tb.pcat(P, sdim=sdim, fdim=fdim, isnorm=True, eigbkd='svd')

    Ug = Ug[..., :npcs]
    Up = Up[..., :npcs]

    return cossim(Ug, Up, mode=mode, cdim=None, dim=-2, keepdim=keepdim, reduction=reduction)


if __name__ == '__main__':

    dim = (-2, -1)
    # dim = None

    print('---cossim')
    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = cossim(P, G, cdim=None, dim=dim, reduction=None)
    C2 = cossim(P, G, cdim=None, dim=dim, reduction='sum')
    C3 = cossim(P, G, cdim=None, dim=dim, reduction='mean')
    print(C1, C1.shape, C2, C3)

    # complex in real format
    C1 = cossim(P, G, cdim=1, dim=dim, reduction=None)
    C2 = cossim(P, G, cdim=1, dim=dim, reduction='sum')
    C3 = cossim(P, G, cdim=1, dim=dim, reduction='mean')
    print(C1, C1.shape, C2, C3)

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = cossim(P, G, cdim=None, dim=dim, reduction=None)
    C2 = cossim(P, G, cdim=None, dim=dim, reduction='sum')
    C3 = cossim(P, G, cdim=None, dim=dim, reduction='mean')
    print(C1, C1.shape, C2, C3)

    print('---peacor')
    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = peacor(P, G, cdim=None, dim=dim, reduction=None)
    C2 = peacor(P, G, cdim=None, dim=dim, reduction='sum')
    C3 = peacor(P, G, cdim=None, dim=dim, reduction='mean')
    print(C1, C2, C3)

    # complex in real format
    C1 = peacor(P, G, cdim=1, dim=dim, reduction=None)
    C2 = peacor(P, G, cdim=1, dim=dim, reduction='sum')
    C3 = peacor(P, G, cdim=1, dim=dim, reduction='mean')
    print(C1, C2, C3)

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = peacor(P, G, cdim=None, dim=dim, reduction=None)
    C2 = peacor(P, G, cdim=None, dim=dim, reduction='sum')
    C3 = peacor(P, G, cdim=None, dim=dim, reduction='mean')
    print(C1, C2, C3)

    print('---compare pearson')
    x = th.randn(2, 3) + 1j*th.randn(2, 3)
    print(th.corrcoef(x))
    print(peacor(x[0], x[1]))
    print(peacor(x[1], x[0]))

    print('---compare eigen vector correlation (complex in real)')
    G = th.randn(2, 3, 2, 64, 4)
    P = th.randn(2, 3, 2, 64, 4)
    print(eigveccor(G, G, npcs=4, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))
    print(eigveccor(G, G, npcs=4, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction=None).shape)
    
    print('---compare eigen vector correlation (complex in complex)')
    G = th.randn(2, 3, 64, 4) + 1j*th.randn(2, 3, 64, 4)
    P = th.randn(2, 3, 64, 4) + 1j*th.randn(2, 3, 64, 4)
    print(eigveccor(G, G, npcs=4, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))
    print(eigveccor(G, G, npcs=4, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction=None).shape)
