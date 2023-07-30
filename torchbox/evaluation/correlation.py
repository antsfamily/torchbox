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
       s = \frac{<{\bf p}, {\bf g}>}{\|p\|_2\|g\|_2}

    .. note:: 
       For complex, the magnitude still gives the "similarity" between them, 
       where the complex angle gives the complex phase factor required to fully reach that similarity. 
       refers `Cosine similarity between complex vectors <https://math.stackexchange.com/questions/273527/cosine-similarity-between-complex-vectors#:~:text=For%20complex%2C%20the%20magnitude%20still%20gives%20the%20%22similarity%22,is%20a%20real%20scalar%20multiple%20of%20y%20y.>`_

    Parameters
    ----------
    P : tensor
        the first/left input, such as the predicted
    G : tensor
        the second/right input, such as the ground-truth
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : tuple, None, optional
        The dimension axis for computing cosine similarity. The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    S : scalar or tensor
        The cosine similarity value of inputs.
    
    Examples
    --------

    ::

        import torch as th
        from torchbox import cossim

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)
        dim = (-2, -1)

        # real
        C1 = cossim(X, Y, cdim=None, dim=dim, reduction=None)
        C2 = cossim(X, Y, cdim=None, dim=dim, reduction='sum')
        C3 = cossim(X, Y, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = cossim(X, Y, cdim=1, dim=dim, reduction=None)
        C2 = cossim(X, Y, cdim=1, dim=dim, reduction='sum')
        C3 = cossim(X, Y, cdim=1, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = cossim(X, Y, cdim=None, dim=dim, reduction=None)
        C2 = cossim(X, Y, cdim=None, dim=dim, reduction='sum')
        C3 = cossim(X, Y, cdim=None, dim=dim, reduction='mean')
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

    sdim = tb.rdcdim(S.ndim, cdim=cdim, dim=dim, keepcdim=True, reduction=reduction)

    return tb.reduce(S, dim=sdim, keepdim=keepdim, reduction=reduction)

def peacor(P, G, mode=None, cdim=None, dim=None, keepdim=False, reduction=None):
    r"""compute the Pearson Correlation Coefficient

    The Pearson correlation coefficient can be viewed as the cosine similarity of centered (remove mean) input.

    Parameters
    ----------
    P : tensor
        the first/left input, such as the predicted
    G : tensor
        the second/right input, such as the ground-truth
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
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
    
    Examples
    --------

    ::

        import torch as th
        from torchbox import peacor

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)
        dim = (-2, -1)

        # real
        C1 = peacor(X, Y, cdim=None, dim=dim, reduction=None)
        C2 = peacor(X, Y, cdim=None, dim=dim, reduction='sum')
        C3 = peacor(X, Y, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = peacor(X, Y, cdim=1, dim=dim, reduction=None)
        C2 = peacor(X, Y, cdim=1, dim=dim, reduction='sum')
        C3 = peacor(X, Y, cdim=1, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = peacor(X, Y, cdim=None, dim=dim, reduction=None)
        C2 = peacor(X, Y, cdim=None, dim=dim, reduction='sum')
        C3 = peacor(X, Y, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        x = th.randn(2, 3) + 1j*th.randn(2, 3)
        print(th.corrcoef(x))
        print(peacor(x[0], x[1]))
        print(peacor(x[1], x[0]))

    """

    P = P - tb.mean(P, cdim=cdim, dim=dim, keepdim=True)
    G = G - tb.mean(G, cdim=cdim, dim=dim, keepdim=True)

    return cossim(G, P, mode=mode, cdim=cdim, dim=dim, keepdim=keepdim, reduction=reduction)


if __name__ == '__main__':

    dim = (-2, -1)
    # dim = None

    print('---cossim')
    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)
    Y = th.randn(5, 2, 3, 4)

    # real
    C1 = cossim(X, Y, cdim=None, dim=dim, reduction=None)
    C2 = cossim(X, Y, cdim=None, dim=dim, reduction='sum')
    C3 = cossim(X, Y, cdim=None, dim=dim, reduction='mean')
    print(C1, C2, C3)

    # complex in real format
    C1 = cossim(X, Y, cdim=1, dim=dim, reduction=None)
    C2 = cossim(X, Y, cdim=1, dim=dim, reduction='sum')
    C3 = cossim(X, Y, cdim=1, dim=dim, reduction='mean')
    print(C1, C2, C3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
    C1 = cossim(X, Y, cdim=None, dim=dim, reduction=None)
    C2 = cossim(X, Y, cdim=None, dim=dim, reduction='sum')
    C3 = cossim(X, Y, cdim=None, dim=dim, reduction='mean')
    print(C1, C2, C3)

    print('---peacor')
    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)
    Y = th.randn(5, 2, 3, 4)

    # real
    C1 = peacor(X, Y, cdim=None, dim=dim, reduction=None)
    C2 = peacor(X, Y, cdim=None, dim=dim, reduction='sum')
    C3 = peacor(X, Y, cdim=None, dim=dim, reduction='mean')
    print(C1, C2, C3)

    # complex in real format
    C1 = peacor(X, Y, cdim=1, dim=dim, reduction=None)
    C2 = peacor(X, Y, cdim=1, dim=dim, reduction='sum')
    C3 = peacor(X, Y, cdim=1, dim=dim, reduction='mean')
    print(C1, C2, C3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
    C1 = peacor(X, Y, cdim=None, dim=dim, reduction=None)
    C2 = peacor(X, Y, cdim=None, dim=dim, reduction='sum')
    C3 = peacor(X, Y, cdim=None, dim=dim, reduction='mean')
    print(C1, C2, C3)

    print('---compare pearson')
    x = th.randn(2, 3) + 1j*th.randn(2, 3)
    print(th.corrcoef(x))
    print(peacor(x[0], x[1]))
    print(peacor(x[1], x[0]))
