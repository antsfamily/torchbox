#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : error.py
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


def mse(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the mean square error

    Both complex and real representation are supported.

    .. math::
       {\rm MSE}({\bf P, G}) = \frac{1}{N}\|{\bf P} - {\bf G}\|_2^2 = \frac{1}{N}\sum_{i=1}^N(|p_i - g_i|)^2

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         mean square error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = mse(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = mse(P, G, cdim=1, dim=(-2, -1), reduction=None)
        C2 = mse(P, G, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = mse(P, G, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = mse(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

    if P.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        P = P.to(th.float64)
    if G.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        G = G.to(th.float64)

    E = th.mean(tb.pow(P - G, cdim=cdim, keepdim=True), dim=dim, keepdim=True)

    sdim = tb.dimreduce(E.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(E, dim=sdim, keepdim=keepdim, reduction=reduction)


def sse(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the sum square error

    Both complex and real representation are supported.

    .. math::
       {\rm SSE}({\bf P, G}) = \|{\bf P} - {\bf G}\|_2^2 = \sum_{i=1}^N(|p_i - g_i|)^2

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         sum square error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = sse(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = sse(P, G, cdim=1, dim=(-2, -1), reduction=None)
        C2 = sse(P, G, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = sse(P, G, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = sse(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

    if P.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        P = P.to(th.float64)
    if G.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        G = G.to(th.float64)

    E = th.sum(tb.pow(P - G, cdim=cdim, keepdim=True), dim=dim, keepdim=True)

    sdim = tb.dimreduce(E.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(E, dim=sdim, keepdim=keepdim, reduction=reduction)


def mae(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the mean absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm MAE}({\bf P, G}) = \frac{1}{N}|{\bf P} - {\bf G}| = \frac{1}{N}\sum_{i=1}^N |p_i - g_i|

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         mean absoluted error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = mae(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = mae(P, G, cdim=1, dim=(-2, -1), reduction=None)
        C2 = mae(P, G, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = mae(P, G, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = mae(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

    if P.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        P = P.to(th.float64)
    if G.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        G = G.to(th.float64)

    E = th.mean(tb.abs(P - G, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    
    sdim = tb.dimreduce(E.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(E, dim=sdim, keepdim=keepdim, reduction=reduction)


def sae(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the sum absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm SAE}({\bf P, G}) = |{\bf P} - {\bf G}| = \sum_{i=1}^N |p_i - g_i|

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
        sum absoluted error

    Examples
    ---------

    ::
    
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = sae(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = sae(P, G, cdim=1, dim=(-2, -1), reduction=None)
        C2 = sae(P, G, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = sae(P, G, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = sae(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

    if P.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        P = P.to(th.float64)
    if G.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        G = G.to(th.float64)

    E = th.sum(tb.abs(P - G, cdim=cdim, keepdim=True), dim=dim, keepdim=True)

    sdim = tb.dimreduce(E.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(E, dim=sdim, keepdim=keepdim, reduction=reduction)

def nmse(P, G, mode='Gpowsum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized mean square error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    mode : str
        mode of normalization
        ``'Gpowsum'`` (default) normalized square error with the power summation of :attr:`G`, 
        ``'Gabssum'`` (default) normalized square error with the amplitude summation of :attr:`G`, 
        ``'Gpowmax'`` normalized square error with the maximum power of :attr:`G`,
        ``'Gabsmax'`` normalized square error with the maximum amplitude of :attr:`G`,
        ``'GpeakV'`` normalized square error with the square of peak value (V) of :attr:`G`;
        ``'Gfnorm'`` normalized square error with Frobenius norm of :attr:`G`;
        ``'Gpnorm'`` normalized square error with p-norm of :attr:`G`;
        ``'fnorm'`` normalized :attr:`P` and :attr:`G` with Frobenius norm,
        ``'pnormV'`` normalized :attr:`P` and :attr:`G` with p-norm, respectively, where V is a float or integer number; 
        ``'zscore'`` normalized :attr:`P` and :attr:`G` with zscore method.
        ``'std'`` normalized :attr:`P` and :attr:`G` with standard deviation.
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
        normalized mean square error

    Examples
    ---------

    ::

        mode = 'Gabssum'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

    if P.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        P = P.to(th.float64)
    if G.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        G = G.to(th.float64)

    if mode.lower() == 'gpowsum':
        E = th.mean(tb.pow(P - G, cdim=cdim, keepdim=True) / tb.pow(G, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gabssum':
        E = th.mean(tb.pow(P - G, cdim=cdim, keepdim=True) / tb.abs(G, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gpowmax':
        E = th.mean(tb.pow(P - G, cdim=cdim, keepdim=True) / tb.pow(G, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gabsmax':
        E = th.mean(tb.pow(P - G, cdim=cdim, keepdim=True) / tb.abs(G, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif 'gpeak' in mode.lower():
        p = tb.str2num(mode.lower(), float)
        p = 1 if len(p) == 0 else p[0]
        E = th.mean(tb.pow(P - G, cdim=cdim, keepdim=True) / p, dim=dim, keepdim=True)
    elif 'gfnorm' == mode.lower():
        E = tb.pow(P - G, cdim=cdim, keepdim=True) / tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True)
        E = th.mean(E, dim=dim, keepdim=True)
    elif 'gpnorm' == mode[:6].lower():
        p = tb.str2num(mode.lower(), float)
        p = 2. if len(p) == 0 else p[0]
        E = tb.pow(P - G, cdim=cdim, keepdim=True) / tb.norm(G, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True)
        E = th.mean(E, dim=dim, keepdim=True)
    elif 'fnorm' == mode.lower():
        E = th.mean(tb.pow(P / tb.norm(P, mode='fro', cdim=cdim, dim=dim, keepdim=True) - G / tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'pnorm' == mode[:5].lower():
        p = tb.str2num(mode.lower(), float)
        p = 2 if len(p) == 0 else p[0]
        E = th.mean(tb.pow(P / tb.norm(P, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True) - G / tb.norm(G, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'zscore' == mode.lower():
        E = tb.zscore(P, cdim=cdim, dim=dim, retall=False) - tb.zscore(G, cdim=cdim, dim=dim, retall=False)
        E = th.mean(tb.pow(E, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'std' == mode.lower():
        E = P / (tb.std(P, cdim=cdim, dim=dim, keepdim=True) + tb.EPS) - G / (tb.std(G, cdim=cdim, dim=dim, keepdim=True) + tb.EPS)
        E = th.mean(tb.pow(E, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    else:
        raise ValueError('Not supported mode: %s' % mode)

    sdim = tb.dimreduce(E.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(E, dim=sdim, keepdim=keepdim, reduction=reduction)


def nsse(P, G, mode='Gpowsum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized sum square error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    mode : str
        mode of normalization, 
        ``'Gpowsum'`` (default) normalized square error with the power summation of :attr:`G`, 
        ``'Gabssum'`` (default) normalized square error with the amplitude summation of :attr:`G`, 
        ``'Gpowmax'`` normalized square error with the maximum power of :attr:`G`,
        ``'Gabsmax'`` normalized square error with the maximum amplitude of :attr:`G`,
        ``'GpeakV'`` normalized square error with the square of peak value (V) of :attr:`G`;
        ``'Gfnorm'`` normalized square error with Frobenius norm of :attr:`G`;
        ``'Gpnorm'`` normalized square error with p-norm of :attr:`G`;
        ``'fnorm'`` normalized :attr:`P` and :attr:`G` with Frobenius norm,
        ``'pnormV'`` normalized :attr:`P` and :attr:`G` with p-norm, respectively, where V is a float or integer number; 
        ``'zscore'`` normalized :attr:`P` and :attr:`G` with zscore method.
        ``'std'`` normalized :attr:`P` and :attr:`G` with standard deviation.
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
        normalized sum square error

    Examples
    ---------

    ::

        mode = 'Gabssum'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

    if P.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        P = P.to(th.float64)
    if G.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        G = G.to(th.float64)
    
    if mode.lower() == 'gpowsum':
        E = th.sum(tb.pow(P - G, cdim=cdim, keepdim=True) / tb.pow(G, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gabssum':
        E = th.sum(tb.pow(P - G, cdim=cdim, keepdim=True) / tb.abs(G, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gpowmax':
        E = th.sum(tb.pow(P - G, cdim=cdim, keepdim=True) / tb.pow(G, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gabsmax':
        E = th.sum(tb.pow(P - G, cdim=cdim, keepdim=True) / tb.abs(G, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif 'gpeak' in mode.lower():
        p = tb.str2num(mode.lower(), float)
        p = 1. if len(p) == 0 else p[0]
        E = th.sum(tb.pow(P - G, cdim=cdim, keepdim=True) / p, dim=dim, keepdim=True)
    elif 'gfnorm' == mode.lower():
        E = tb.pow(P - G, cdim=cdim, keepdim=True) / tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True)
        E = th.sum(E if keepdim or (cdim is None) else E.squeeze(cdim), dim=dim, keepdim=True)
    elif 'gpnorm' == mode[:6].lower():
        p = tb.str2num(mode.lower(), float)
        p = 2 if len(p) == 0 else p[0]
        E = tb.pow(P - G, cdim=cdim, keepdim=True) / tb.norm(G, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True)
        E = th.sum(E if keepdim or (cdim is None) else E.squeeze(cdim), dim=dim, keepdim=True)
    elif 'fnorm' == mode.lower():
        E = th.sum(tb.pow(P / tb.norm(P, mode='fro', cdim=cdim, dim=dim, keepdim=True) - G / tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'pnorm' == mode[:5].lower():
        p = tb.str2num(mode.lower(), float)
        p = 2 if len(p) == 0 else p[0]
        E = th.sum(tb.pow(P / tb.norm(P, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True) - G / tb.norm(G, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'zscore' == mode.lower():
        E = tb.zscore(P, cdim=cdim, dim=dim, retall=False) - tb.zscore(G, cdim=cdim, dim=dim, retall=False)
        E = th.sum(tb.pow(E, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'std' == mode.lower():
        E = P / (tb.std(P, cdim=cdim, dim=dim, keepdim=True) + tb.EPS) - G / (tb.std(G, cdim=cdim, dim=dim, keepdim=True) + tb.EPS)
        E = th.sum(tb.pow(E, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    else:
        raise ValueError('Not supported mode: %s' % mode)

    sdim = tb.dimreduce(E.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(E, dim=sdim, keepdim=keepdim, reduction=reduction)


def nmae(P, G, mode='Gabssum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized mean absoluted error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    mode : str
        mode of normalization, 
        ``'Gabssum'`` (default) normalized square error with the amplitude summation of :attr:`G`, 
        ``'Gpowsum'`` normalized square error with the power summation of :attr:`G`, 
        ``'Gabsmax'`` normalized square error with the maximum amplitude of :attr:`G`,
        ``'Gpowmax'`` normalized square error with the maximum power of :attr:`G`,
        ``'GpeakV'`` normalized square error with the square of peak value (V) of :attr:`G`;
        ``'Gfnorm'`` normalized square error with Frobenius norm of :attr:`G`;
        ``'Gpnorm'`` normalized square error with p-norm of :attr:`G`;
        ``'fnorm'`` normalized :attr:`P` and :attr:`G` with Frobenius norm,
        ``'pnormV'`` normalized :attr:`P` and :attr:`G` with p-norm, respectively, where V is a float or integer number; 
        ``'zscore'`` normalized :attr:`P` and :attr:`G` with zscore method.
        ``'std'`` normalized :attr:`P` and :attr:`G` with standard deviation.
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         normalized mean absoluted error

    Examples
    ---------

    ::

        mode = 'Gabssum'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)
  
    """

    if P.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        P = P.to(th.float64)
    if G.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        G = G.to(th.float64)

    if mode.lower() == 'gabssum':
        E = th.mean(tb.abs(P - G, cdim=cdim, keepdim=True) / tb.abs(G, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gpowsum':
        E = th.mean(tb.abs(P - G, cdim=cdim, keepdim=True) / tb.pow(G, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gabsmax':
        E = th.mean(tb.abs(P - G, cdim=cdim, keepdim=True) / tb.abs(G, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gpowmax':
        E = th.mean(tb.abs(P - G, cdim=cdim, keepdim=True) / tb.pow(G, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif 'gpeak' in mode.lower():
        p = tb.str2num(mode.lower(), float)
        p = 1. if len(p) == 0 else p[0]
        E = th.mean(tb.abs(P - G, cdim=cdim, keepdim=True) / p, dim=dim, keepdim=True)
    elif 'gfnorm' == mode.lower():
        E = tb.abs(P - G, cdim=cdim, keepdim=True) / tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True)
        E = th.mean(E if keepdim or (cdim is None) else E.squeeze(cdim), dim=dim, keepdim=True)
    elif 'gpnorm' == mode[:6].lower():
        p = tb.str2num(mode.lower(), float)
        p = 2 if len(p) == 0 else p[0]
        E = tb.abs(P - G, cdim=cdim, keepdim=True) / tb.norm(G, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True)
        E = th.mean(E if keepdim or (cdim is None) else E.squeeze(cdim), dim=dim, keepdim=True)
    elif 'fnorm' == mode.lower():
        E = th.mean(tb.abs(P / tb.norm(P, mode='fro', cdim=cdim, dim=dim, keepdim=True) - G / tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'pnorm' == mode[:5].lower():
        p = tb.str2num(mode.lower(), float)
        p = 2 if len(p) == 0 else p[0]
        E = th.mean(tb.abs(P / tb.norm(P, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True) - G / tb.norm(G, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'zscore' == mode.lower():
        E = tb.zscore(P, cdim=cdim, dim=dim, retall=False) - tb.zscore(G, cdim=cdim, dim=dim, retall=False)
        E = th.mean(tb.abs(E, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'std' == mode.lower():
        E = P / (tb.std(P, cdim=cdim, dim=dim, keepdim=True) + tb.EPS) - G / (tb.std(G, cdim=cdim, dim=dim, keepdim=True) + tb.EPS)
        E = th.mean(tb.abs(E, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    else:
        raise ValueError('Not supported mode: %s' % mode)

    sdim = tb.dimreduce(E.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(E, dim=sdim, keepdim=keepdim, reduction=reduction)


def nsae(P, G, mode='Gabssum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized sum absoluted error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    mode : str
        mode of normalization, 
        ``'Gabssum'`` (default) normalized square error with the amplitude summation of :attr:`G`, 
        ``'Gpowsum'`` normalized square error with the power summation of :attr:`G`, 
        ``'Gabsmax'`` normalized square error with the maximum amplitude of :attr:`G`,
        ``'Gpowmax'`` normalized square error with the maximum power of :attr:`G`,
        ``'GpeakV'`` normalized square error with the square of peak value (V) of :attr:`G`;
        ``'Gfnorm'`` normalized square error with Frobenius norm of :attr:`G`;
        ``'Gpnorm'`` normalized square error with p-norm of :attr:`G`;
        ``'fnorm'`` normalized :attr:`P` and :attr:`G` with Frobenius norm,
        ``'pnormV'`` normalized :attr:`P` and :attr:`G` with p-norm, respectively, where V is a float or integer number; 
        ``'zscore'`` normalized :attr:`P` and :attr:`G` with zscore method.
        ``'std'`` normalized :attr:`P` and :attr:`G` with standard deviation.
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         sum absoluted error

    Examples
    ---------

    ::

        mode = 'Gabssum'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

    if P.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        P = P.to(th.float64)
    if G.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        G = G.to(th.float64)

    if mode.lower() == 'gabssum':
        E = th.sum(tb.abs(P - G, cdim=cdim, keepdim=True) / tb.abs(G, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gpowsum':
        E = th.sum(tb.abs(P - G, cdim=cdim, keepdim=True) / tb.pow(G, cdim=cdim, keepdim=True).sum(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gabsmax':
        E = th.sum(tb.abs(P - G, cdim=cdim, keepdim=True) / tb.abs(G, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif mode.lower() == 'gpowmax':
        E = th.sum(tb.abs(P - G, cdim=cdim, keepdim=True) / tb.pow(G, cdim=cdim, keepdim=True).amax(dim=dim, keepdim=True), dim=dim, keepdim=True)
    elif 'gpeak' in mode.lower():
        p = tb.str2num(mode.lower(), float)
        p = 1. if len(p) == 0 else p[0]
        E = th.sum(tb.abs(P - G, cdim=cdim, keepdim=True) / p, dim=dim, keepdim=True)
    elif 'gfnorm' == mode.lower():
        E = tb.abs(P - G, cdim=cdim, keepdim=True) / tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True)
        E = th.sum(E if keepdim or (cdim is None) else E.squeeze(cdim), dim=dim, keepdim=True)
    elif 'gpnorm' == mode[:6].lower():
        p = tb.str2num(mode.lower(), float)
        p = 2 if len(p) == 0 else p[0]
        E = tb.abs(P - G, cdim=cdim, keepdim=True) / tb.norm(G, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True)
        E = th.sum(E if keepdim or (cdim is None) else E.squeeze(cdim), dim=dim, keepdim=True)
    elif 'fnorm' == mode.lower():
        E = th.sum(tb.abs(P / tb.norm(P, mode='fro', cdim=cdim, dim=dim, keepdim=True) - G / tb.norm(G, mode='fro', cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'pnorm' == mode[:5].lower():
        p = tb.str2num(mode.lower(), float)
        p = 2 if len(p) == 0 else p[0]
        E = th.sum(tb.abs(P / tb.norm(P, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True) - G / tb.norm(G, mode='p'+str(p), cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'zscore' == mode.lower():
        E = tb.zscore(P, cdim=cdim, dim=dim, retall=False) - tb.zscore(G, cdim=cdim, dim=dim, retall=False)
        E = th.sum(tb.abs(E, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    elif 'std' == mode.lower():
        E = P / (tb.std(P, cdim=cdim, dim=dim, keepdim=True) + tb.EPS) - G / (tb.std(G, cdim=cdim, dim=dim, keepdim=True) + tb.EPS)
        E = th.sum(tb.abs(E, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    else:
        raise ValueError('Not supported mode: %s' % mode)

    sdim = tb.dimreduce(E.ndim, cdim=cdim, dim=dim, keepcdim=False, reduction=reduction)

    return tb.reduce(E, dim=sdim, keepdim=keepdim, reduction=reduction)


if __name__ == '__main__':

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = mse(P, G, cdim=None, dim=(-2, -1), reduction=None)
    C2 = mse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = mse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'mse')

    # complex in real format
    C1 = mse(P, G, cdim=1, dim=(-2, -1), reduction=None)
    C2 = mse(P, G, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = mse(P, G, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'mse')

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = mse(P, G, cdim=None, dim=(-2, -1), reduction=None)
    C2 = mse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = mse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'mse')

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = sse(P, G, cdim=None, dim=(-2, -1), reduction=None)
    C2 = sse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = sse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'sse')

    # complex in real format
    C1 = sse(P, G, cdim=1, dim=(-2, -1), reduction=None)
    C2 = sse(P, G, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = sse(P, G, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'sse')

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = sse(P, G, cdim=None, dim=(-2, -1), reduction=None)
    C2 = sse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = sse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'sse')

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = mae(P, G, cdim=None, dim=(-2, -1), reduction=None)
    C2 = mae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = mae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'mae')

    # complex in real format
    C1 = mae(P, G, cdim=1, dim=(-2, -1), reduction=None)
    C2 = mae(P, G, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = mae(P, G, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'mae')

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = mae(P, G, cdim=None, dim=(-2, -1), reduction=None)
    C2 = mae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = mae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'mae')

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = sae(P, G, cdim=None, dim=(-2, -1), reduction=None)
    C2 = sae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = sae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'sae')

    # complex in real format
    C1 = sae(P, G, cdim=1, dim=(-2, -1), reduction=None)
    C2 = sae(P, G, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = sae(P, G, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'sae')

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = sae(P, G, cdim=None, dim=(-2, -1), reduction=None)
    C2 = sae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = sae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'sae')

    # ----------------------------------------------------------------
    print('-----------normalized')
    
    mode = 'fnorm'
    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
    C2 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nmse')

    # complex in real format
    C1 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
    C2 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nmse')

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
    C2 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nmse')

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
    C2 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nsse')

    # complex in real format
    C1 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
    C2 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nsse')

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
    C2 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nsse')

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
    C2 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nmae')

    # complex in real format
    C1 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
    C2 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nmae')

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
    C2 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nmae')

    th.manual_seed(2020)
    P = th.randn(5, 2, 3, 4)
    G = th.randn(5, 2, 3, 4)

    # real
    C1 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
    C2 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nsae')

    # complex in real format
    C1 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
    C2 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nsae')

    # complex in complex format
    P = P[:, 0, ...] + 1j * P[:, 1, ...]
    G = G[:, 0, ...] + 1j * G[:, 1, ...]
    C1 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
    C2 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3, 'nsae')
