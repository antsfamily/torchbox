#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : norm.py
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


def fnorm(X, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""obtain the f-norm of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm norm}({\bf X}) = \|{\bf X}\|_2 = \left(\sum_{x_i\in {\bf X}}|x_i|^2\right)^{\frac{1}{2}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, None or optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    tensor
         the inputs's f-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)

        print('---norm')
        # real
        C1 = fnorm(X, cdim=None, dim=(-2, -1), reduction=None)
        C2 = fnorm(X, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = fnorm(X, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = fnorm(X, cdim=1, dim=(-2, -1), reduction=None)
        C2 = fnorm(X, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = fnorm(X, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        C1 = fnorm(X, cdim=None, dim=(-2, -1), reduction=None)
        C2 = fnorm(X, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = fnorm(X, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # ---output
        ---norm
        tensor([[2.8719, 2.8263],
                [3.1785, 3.4701],
                [4.6697, 3.2955],
                [3.0992, 2.6447],
                [3.5341, 3.5779]]) tensor(33.1679) tensor(3.3168)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
    """

    if X.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        X = X.to(th.float64)

    if th.is_complex(X):  # complex in complex
        if dim is None:
            F = (X.real*X.real + X.imag*X.imag).sum().sqrt()
        else:
            F = (X.real*X.real + X.imag*X.imag).sum(dim=dim).sqrt()
    else:
        if cdim is None:  # real
            if dim is None:
                F = (X**2).sum().sqrt()
            else:
                F = (X**2).sum(dim=dim).sqrt()
        else:  # complex in real
            if dim is None:
                F = (X**2).sum(dim=cdim).sum().sqrt()
            else:
                F = (X**2).sum(dim=cdim, keepdims=keepcdim).sum(dim=dim).sqrt()

    if reduction in ['sum', 'SUM']:
        F = th.sum(F)
    if reduction in ['mean', 'MEAN']:
        F = th.mean(F)
    
    return F


def pnorm(X, cdim=None, dim=None, keepcdim=False, p=2, reduction='mean'):
    r"""obtain the p-norm of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm pnorm}({\bf X}) = \|{\bf X}\|_p = \left(\sum_{x_i\in {\bf X}}|x_i|^p\right)^{\frac{1}{p}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    p : int
        Specifies the power. The default is 2.
    reduction : str, None or optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    tensor
         the inputs's p-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        
        print('---pnorm')
        # real
        C1 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction=None)
        C2 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction='sum')
        C3 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = pnorm(X, cdim=1, dim=(-2, -1), p=2, reduction=None)
        C2 = pnorm(X, cdim=1, dim=(-2, -1), p=2, reduction='sum')
        C3 = pnorm(X, cdim=1, dim=(-2, -1), p=2, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        C1 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction=None)
        C2 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction='sum')
        C3 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction='mean')
        print(C1, C2, C3)

        # ---output
        ---pnorm
        tensor([[2.8719, 2.8263],
                [3.1785, 3.4701],
                [4.6697, 3.2955],
                [3.0992, 2.6447],
                [3.5341, 3.5779]]) tensor(33.1679) tensor(3.3168)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
    """

    if X.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        X = X.to(th.float64)

    if th.is_complex(X):  # complex in complex
        if dim is None:
            F = (X.abs().pow(p).sum()).pow(1/p)
        else:
            F = (X.abs().pow(p).sum(dim=dim)).pow(1/p)
    else:
        if cdim is None:  # real
            if dim is None:
                F = (X.abs().pow(p).sum()).pow(1/p)
            else:
                F = (X.abs().pow(p).sum(dim=dim)).pow(1/p)
        else:  # complex in real
            if dim is None:
                F = (X**2).sum(dim=cdim).sqrt().pow(p).sum().pow(1/p)
            else:
                F = (X**2).sum(dim=cdim, keepdims=keepcdim).sqrt().pow(p).sum(dim=dim).pow(1/p)
            
    if reduction in ['sum', 'SUM']:
        F = th.sum(F)
    if reduction in ['mean', 'MEAN']:
        F = th.mean(F)
    
    return F


if __name__ == '__main__':

    x = th.tensor([1 ,2, 3])
    print(fnorm(x))

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)

    print('---norm')
    # real
    C1 = fnorm(X, cdim=None, dim=(-2, -1), reduction=None)
    C2 = fnorm(X, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = fnorm(X, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3)

    # complex in real format
    C1 = fnorm(X, cdim=1, dim=(-2, -1), reduction=None)
    C2 = fnorm(X, cdim=1, dim=(-2, -1), reduction='sum')
    C3 = fnorm(X, cdim=1, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    C1 = fnorm(X, cdim=None, dim=(-2, -1), reduction=None)
    C2 = fnorm(X, cdim=None, dim=(-2, -1), reduction='sum')
    C3 = fnorm(X, cdim=None, dim=(-2, -1), reduction='mean')
    print(C1, C2, C3)

    th.manual_seed(2020)
    X = th.randn(5, 2, 3, 4)

    print('---pnorm')
    # real
    C1 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction=None)
    C2 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction='sum')
    C3 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction='mean')
    print(C1, C2, C3)

    # complex in real format
    C1 = pnorm(X, cdim=1, dim=(-2, -1), p=2, reduction=None)
    C2 = pnorm(X, cdim=1, dim=(-2, -1), p=2, reduction='sum')
    C3 = pnorm(X, cdim=1, dim=(-2, -1), p=2, reduction='mean')
    print(C1, C2, C3)

    # complex in complex format
    X = X[:, 0, ...] + 1j * X[:, 1, ...]
    C1 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction=None)
    C2 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction='sum')
    C3 = pnorm(X, cdim=None, dim=(-2, -1), p=2, reduction='mean')
    print(C1, C2, C3)