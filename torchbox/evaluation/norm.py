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


def norm(X, mode='2', cdim=None, dim=None, keepdim=False, reduction=None):
    r"""obtain the norm of a tensor

    Both complex and real representation are supported.

    F-norm (Frobenius):

    .. math::
       \|{\bf X}\|_F = \|{\bf X}\|_p = \left(\sum_{x_i\in {\bf X}}|x_i|^2\right)^{\frac{1}{2}}
    
    p-norm:

    .. math::
       \|{\bf X}\|_p = \|{\bf X}\|_p = \left(\sum_{x_i\in {\bf X}}|x_i|^p\right)^{\frac{1}{p}}

    2-norm or spectral norm:

    .. math::
       \|{\bf X}\|_2 = \sqrt{\lambda_1} = \sqrt{{\rm max} {\lambda({\bf X}^H{\bf X})}}

    1-norm:
    
    .. math::
       \|{\bf X}\|_1 = {\rm max}\sum_{i=1}^M|x_ij|

       
    Parameters
    ----------
    X : Tensor
        input
    mode : str
        the mode of norm. ``'2'`` means 2-norm (default), ``'1'`` means 1-norm, ``'px'`` means p-norm (x is the power), 
        ``'fro'`` means Frobenius-norm  The default is ``'2'``.
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. For 2-norm, :attr:`dim` must be specified. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    tensor
         the inputs's p-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X, cdim = th.randn(5, 2, 3, 4), 1
        X, cdim = th.randn(2, 3, 4), 0

        # real
        C1 = norm(X, mode='fro', cdim=None, dim=(-2, -1), keepdim=False)
        C2 = norm(X, mode='2', cdim=None, dim=(-2, -1), keepdim=False)
        C3 = norm(X, mode='1', cdim=None, dim=-1, keepdim=False)
        C4 = norm(X, mode='p1', cdim=None, dim=(-2, -1), keepdim=False)
        C5 = norm(X, mode='p2', cdim=None, dim=(-2, -1), keepdim=False)
        print(C1, C2, C3, C4, C5)

        # complex in real format
        C1 = norm(X, mode='fro', cdim=cdim, dim=(-2, -1), keepdim=False)
        C2 = norm(X, mode='2', cdim=cdim, dim=(-2, -1), keepdim=False)
        C3 = norm(X, mode='1', cdim=cdim, dim=-1, keepdim=False)
        C4 = norm(X, mode='p1', cdim=cdim, dim=(-2, -1), keepdim=False)
        C5 = norm(X, mode='p2', cdim=cdim, dim=(-2, -1), keepdim=False)
        print(C1, C2, C3, C4, C5)

        # complex in complex format
        X = tb.r2c(X, cdim=cdim, keepdim=False)
        C1 = norm(X, mode='fro', cdim=None, dim=(-2, -1), keepdim=False)
        C2 = norm(X, mode='2', cdim=None, dim=(-2, -1), keepdim=False)
        C3 = norm(X, mode='1', cdim=None, dim=-1, keepdim=False)
        C4 = norm(X, mode='p1', cdim=None, dim=(-2, -1), keepdim=False)
        C5 = norm(X, mode='p2', cdim=None, dim=(-2, -1), keepdim=False)
        print(C1, C2, C3, C4, C5)

    """

    if X.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        X = X.to(th.float64)

    if 'fro' in mode.lower():
        X = tb.pow(X, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True).sqrt()
    elif 'p' in mode.lower():
        p = tb.str2num(mode, vfn=float)
        p = 2 if len(p) == 0 else p[0]
        X = tb.abs(X, cdim=cdim, keepdim=True).pow(p).sum(dim=dim, keepdim=True).pow(1/p)
    elif '1' == mode:
        X = tb.abs(X, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True)
    elif '2' == mode:
        if dim is None:
            raise ValueError('dim must be specified for 2-norm!')
        lambd = tb.eigvals(tb.matmul(tb.conj(X, cdim=cdim).transpose(*dim), X, cdim=cdim, dim=dim), cdim=cdim, dim=dim, keepdim=True)
        lambd = th.amax(lambd.real, dim=-1, keepdim=True)
        return lambd.unsqueeze(-1) if keepdim else lambd
    else:
        raise ValueError('Not supported mode: %s' % mode)

    sdim = tb.dimreduce(X.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(X, dim=sdim, keepdim=keepdim, reduction=reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    X, cdim = th.randn(5, 2, 3, 4), 1
    # X, cdim = th.randn(2, 3, 4), 0

    # real
    C1 = norm(X, mode='fro', cdim=None, dim=(-2, -1), keepdim=False)
    C2 = norm(X, mode='2', cdim=None, dim=(-2, -1), keepdim=False)
    C3 = norm(X, mode='1', cdim=None, dim=-1, keepdim=False)
    C4 = norm(X, mode='p1', cdim=None, dim=(-2, -1), keepdim=False)
    C5 = norm(X, mode='p2', cdim=None, dim=(-2, -1), keepdim=False)
    print(C1, C2, C3, C4, C5)

    # complex in real format
    C1 = norm(X, mode='fro', cdim=cdim, dim=(-2, -1), keepdim=False)
    C2 = norm(X, mode='2', cdim=cdim, dim=(-2, -1), keepdim=False)
    C3 = norm(X, mode='1', cdim=cdim, dim=-1, keepdim=False)
    C4 = norm(X, mode='p1', cdim=cdim, dim=(-2, -1), keepdim=False)
    C5 = norm(X, mode='p2', cdim=cdim, dim=(-2, -1), keepdim=False)
    print(C1, C2, C3, C4, C5)

    # complex in complex format
    X = tb.r2c(X, cdim=cdim, keepdim=False)
    C1 = norm(X, mode='fro', cdim=None, dim=(-2, -1), keepdim=False)
    C2 = norm(X, mode='2', cdim=None, dim=(-2, -1), keepdim=False)
    C3 = norm(X, mode='1', cdim=None, dim=-1, keepdim=False)
    C4 = norm(X, mode='p1', cdim=None, dim=(-2, -1), keepdim=False)
    C5 = norm(X, mode='p2', cdim=None, dim=(-2, -1), keepdim=False)
    print(C1, C2, C3, C4, C5)
