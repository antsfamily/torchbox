#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : entropy.py
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
from torchbox.utils.const import EPS


def entropy(X, mode='shannon', cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""compute the entropy of the inputs

    .. math::
        {\rm S} = -\sum_{n=0}^N p_i{\rm log}_2 p_n

    where :math:`N` is the number of pixels, :math:`p_n=\frac{|X_n|^2}{\sum_{n=0}^N|X_n|^2}`.

    Parameters
    ----------
    X : tensor
        The complex or real inputs, for complex inputs, both complex and real representations are surpported.
    mode : str, optional
        The entropy mode: ``'shannon'`` or ``'natural'`` (the default is 'shannon')
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued.
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : scalar or tensor
        The entropy of the inputs.
    
    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)

        # real
        S1 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction=None)
        S2 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction='sum')
        S3 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction='mean')
        print(S1, S2, S3)

        # complex in real format
        S1 = entropy(X, mode='shannon', cdim=1, dim=(-2, -1), reduction=None)
        S2 = entropy(X, mode='shannon', cdim=1, dim=(-2, -1), reduction='sum')
        S3 = entropy(X, mode='shannon', cdim=1, dim=(-2, -1), reduction='mean')
        print(S1, S2, S3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        S1 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction=None)
        S2 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction='sum')
        S3 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction='mean')
        print(S1, S2, S3)

        # output
        tensor([[2.5482, 2.7150],
                [2.0556, 2.6142],
                [2.9837, 2.9511],
                [2.4296, 2.7979],
                [2.7287, 2.5560]]) tensor(26.3800) tensor(2.6380)
        tensor([3.2738, 2.5613, 3.2911, 2.7989, 3.2789]) tensor(15.2040) tensor(3.0408)
        tensor([3.2738, 2.5613, 3.2911, 2.7989, 3.2789]) tensor(15.2040) tensor(3.0408)

    """

    if mode in ['Shannon', 'shannon', 'SHANNON']:
        logfunc = th.log2
    if mode in ['Natural', 'natural', 'NATURAL']:
        logfunc = th.log

    if th.is_complex(X):  # complex in complex
        X = X.real*X.real + X.imag*X.imag
    else:
        if cdim is None:  # real
            X = X**2
        else:  # complex in real
            X = th.sum(X**2, dim=cdim, keepdims=keepcdim)

    dim = list(range(X.dim())) if dim is None else dim
    P = th.sum(X, dim=dim, keepdims=True)
    p = X / (P + EPS)
    S = -th.sum(p * logfunc(p + EPS), dim=dim)
    if reduction == 'mean':
        S = th.mean(S)
    if reduction == 'sum':
        S = th.sum(S)

    return S


if __name__ == '__main__':

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)

    # real
    S1 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction=None)
    S2 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction='sum')
    S3 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction='mean')
    print(S1, S2, S3)

    # complex in real format
    S1 = entropy(X, mode='shannon', cdim=1, dim=(-2, -1), reduction=None)
    S2 = entropy(X, mode='shannon', cdim=1, dim=(-2, -1), reduction='sum')
    S3 = entropy(X, mode='shannon', cdim=1, dim=(-2, -1), reduction='mean')
    print(S1, S2, S3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    S1 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction=None)
    S2 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction='sum')
    S3 = entropy(X, mode='shannon', cdim=None, dim=(-2, -1), reduction='mean')
    print(S1, S2, S3)

