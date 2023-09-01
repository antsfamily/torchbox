#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : mathops.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 1.0
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

import numpy as np
import torch as th
import torchbox as tb


def db2mag(db, s=20.):
    r"""Converts decibel values to magnitudes

    .. math::
       {\rm mag} = 10^{db / s}

    Parameters
    ----------
    db : int, float, tuple, list, ndarray, tensor
        The decibel values.
    s : int or float
        The scale values, default is 20.

    Returns
    -------
     int, float, tuple, list, ndarray, tensor
        The magnitudes of inputs with the same type.
    """
    if type(db) is list:
        return list(10 ** (dbi / s) for dbi in db)
    if type(db) is tuple:
        return tuple(10 ** (dbi / s) for dbi in db)
    return 10 ** (db / s)


def mag2db(mag, s=20.):
    r"""Converts decibel values to magnitudes

    .. math::
       {\rm db} = s*{\rm log10}{\rm mag}

    Parameters
    ----------
    mag : int, float, tuple, list, ndarray, tensor
        The magnitude values.
    s : int or float
        The scale values, default is 20.

    Returns
    -------
     int, float, tuple, list, ndarray, tensor
        The decibel of inputs with the same type.
    """
    if type(mag) is th.Tensor:
        return s * th.log10(mag)
    if type(mag) is np.array:
        return s * np.log10(mag)
    if type(mag) is list:
        return list(s * np.log10(magi) for magi in mag)
    if type(mag) is tuple:
        return tuple(s * np.log10(magi) for magi in mag)

    return s * np.log10(mag)


def fnab(n):
    """gives the closest two integer number factor of a number

    Parameters
    ----------
    n : int or float
        the number

    Returns
    -------
    a : int
    b : int
        the factor number

    Examples
    --------

    ::

        print(fnab(5))
        print(fnab(6))
        print(fnab(7))
        print(fnab(8))
        print(fnab(9))

        # ---output
        (2, 3)
        (2, 3)
        (2, 4)
        (2, 4)
        (3, 3)

    """    

    a = int(np.sqrt(n))
    b = a
    while (b * a) < n:
        b += 1
    return a, b

def ebeo(a, b, op='+'):
    r"""element by element operation

    Element by element operation.

    Parameters
    ----------
    a : list, tuple, tensor or array
        The first list/tuple/nparray/tensor.
    b : list, tuple, tensor or array
        The second list/tuple/nparray/tensor.
    op : str, optional
        Supported operations are:
        - ``'+'`` or ``'add'`` for addition (default)
        - ``'-'`` or ``'sub'`` for substraction
        - ``'*'`` or ``'mul'`` for multiplication
        - ``'/'`` or ``'div'`` for division
        - ``'**'`` or ``pow`` for power
        - ``'<'``, or ``'lt'`` for less than
        - ``'<='``, or ``'le'`` for less than or equal to
        - ``'>'``, or ``'gt'`` for greater than
        - ``'>='``, or ``'ge'`` for greater than or equal to
        - ``'&'`` for bitwise and
        - ``'|'`` for bitwise or
        - ``'^'`` for bitwise xor
        - function for custom operation.

    Raises
    ------
    TypeError
        If the specified operator not in the above list, raise a TypeError.
    """
    if op in ['+', 'add']:
        return [i + j for i, j in zip(a, b)]
    if op in ['-', 'sub']:
        return [i - j for i, j in zip(a, b)]
    if op in ['*', 'mul']:
        return [i * j for i, j in zip(a, b)]
    if op in ['/', 'div']:
        return [i / j for i, j in zip(a, b)]
    if op in ['**', '^', 'pow']:
        return [i ** j for i, j in zip(a, b)]
    if isinstance(op, str):
        raise TypeError("Not supported operation: " + op + "!")
    else:
        return [op(i, j) for i, j in zip(a, b)]


def sinc(x):
    """Applies sinc function to a tensor

    Parameters
    ----------
    x : Tensor
        input tensor

    Returns
    -------
    Tensor
        after sinc transformation.
    """

    flag = False
    if type(x) is not th.Tensor:
        flag = True
        x = th.tensor(x)
    x = th.where(x.abs() < 1e-32, th.tensor(1., dtype=x.dtype, device=x.device), th.sin(tb.PI * x) / (tb.PI * x))

    if flag:
        return x.item()
    else:
        return x


def nextpow2(x):
    r"""get the next higher power of 2.

    Given an number :math:`x`, returns the first p such that :math:`2^p >=|x|`. 

    Args:
        x (int or float): an number.

    Returns:
        int: Next higher power of 2.

    Examples:

        ::

            print(prevpow2(-5), nextpow2(-5))
            print(prevpow2(5), nextpow2(5))
            print(prevpow2(0.3), nextpow2(0.3))
            print(prevpow2(7.3), nextpow2(7.3))
            print(prevpow2(-3.5), nextpow2(-3.5))

            # output
            2 3
            2 3
            -2 -1
            2 3
            1 2

    """

    return int(np.ceil(np.log2(np.abs(x) + 1e-32)))


def prevpow2(x):
    r"""get the previous lower power of 2.

    Given an number :math:`x`, returns the first p such that :math:`2^p <=|x|`. 

    Args:
        x (int or float): an number.

    Returns:
        int: Next higher power of 2.

    Examples:

        ::

            print(prevpow2(-5), nextpow2(-5))
            print(prevpow2(5), nextpow2(5))
            print(prevpow2(0.3), nextpow2(0.3))
            print(prevpow2(7.3), nextpow2(7.3))
            print(prevpow2(-3.5), nextpow2(-3.5))

            # output
            2 3
            2 3
            -2 -1
            2 3
            1 2

    """
    
    return int(np.floor(np.log2(np.abs(x) + 1e-32)))


def ematmul(A, B, **kwargs):
    r"""Element-by-element complex multiplication

    like A .* B in matlab

    Parameters
    ----------
    A : Tensor
        any size tensor, both complex and real representation are supported.
        For real representation, the real and imaginary dimension is specified by :attr:`cdim` or :attr:`caxis`.
    B : Tensor
        any size tensor, both complex and real representation are supported.
        For real representation, the real and imaginary dimension is specified by :attr:`cdim` or :attr:`caxis`.
    cdim : int or None, optional
        if :attr:`A` and :attr:`B` are complex tensors but represented in real format, :attr:`cdim` or :attr:`caxis`
        should be specified (Default is :obj:`None`).

    Returns
    -------
    tensor
        result of element-by-element complex multiplication with the same repesentation as :attr:`A` and :attr:`B`.
    
    Examples
    ----------

    ::

        th.manual_seed(2020)
        Ar = th.randn((3, 3, 2))
        Br = th.randn((3, 3, 2))

        Ac = th.view_as_complex(Ar)
        Bc = th.view_as_complex(Br)

        Mr = th.view_as_real(Ac * Bc)
        print(th.sum(Mr - ematmul(Ar, Br, cdim=-1)))
        print(th.sum(Ac * Bc - ematmul(Ac, Bc)))

        # output
        tensor(-1.1921e-07)
        tensor(0.+0.j)

    """

    if 'cdim' in kwargs:
        cdim = kwargs['cdim']
    elif 'caxis' in kwargs:
        cdim = kwargs['caxis']
    else:
        cdim = None

    if th.is_complex(A) or th.is_complex(B):
        return A * B
        # return A.real * B.real - A.imag * B.imag + 1j * (A.real * B.imag + A.imag * B.real)
    elif cdim is None:
        return A * B
    else:
        Nc = A.shape[cdim] // 2
        idxreal, idximag = tb.sl(A.ndim, cdim, slice(None, Nc)), tb.sl(A.ndim, cdim, slice(Nc, None))
        return th.cat((A[idxreal] * B[idxreal] - A[idximag] * B[idximag], A[idxreal] * B[idximag] + A[idximag] * B[idxreal]), dim=cdim)


def matmul(A, B, cdim=None, dim=(-2, -1)):
    r"""Complex matrix multiplication

    like A * B in matlab

    Parameters
    ----------
    A : Tensor
        any size tensor, both complex and real representation are supported.
        For real representation, the real and imaginary dimension is specified by :attr:`cdim` or :attr:`caxis`.
    B : Tensor
        any size tensor, both complex and real representation are supported.
        For real representation, the real and imaginary dimension is specified by :attr:`cdim` or :attr:`caxis`.
    cdim : int or None, optional
        if :attr:`A` and :attr:`B` are complex tensors but represented in real format, :attr:`cdim` or :attr:`caxis`
        should be specified (Default is :obj:`None`).
    dim : tulpe or list
        dimensions for multiplication (default is (-2, -1))

    Returns
    -------
    tensor
        result of complex multiplication with the same repesentation as :attr:`A` and :attr:`B`.
    
    Examples
    ----------

    ::

        th.manual_seed(2020)
        Ar = th.randn((3, 3, 2))
        Br = th.randn((3, 3, 2))

        Ac = th.view_as_complex(Ar)
        Bc = th.view_as_complex(Br)

        print(th.sum(th.matmul(Ac, Bc) - matmul(Ac, Bc)))
        Mr = matmul(Ar, Br, cdim=-1, dim=( 0, 1))
        Mc = th.view_as_real(th.matmul(Ac, Bc))
        print(th.sum(Mr - Mc))

        # output
        tensor(0.+0.j)
        tensor(1.0729e-06)

    """
    newdims = tb.dimpermute(A.ndim, dim=dim, mode='matmul', dir='f')

    A = th.permute(A, newdims)
    B = th.permute(B, newdims)
    if th.is_complex(A) or th.is_complex(B):
        return tb.permute(A @ B, dim=dim, mode='matmul', dir='b')
    elif cdim is None:
        return tb.permute(A @ B, dim=dim, mode='matmul', dir='b')
    else:
        cdim = newdims.index(A.ndim + cdim if cdim < 0 else cdim)
        Nc = A.shape[cdim] // 2
        idxreal, idximag = tb.sl(A.ndim, cdim, slice(None, Nc)), tb.sl(A.ndim, cdim, slice(Nc, None))
        return tb.permute(th.cat((th.matmul(A[idxreal], B[idxreal]) - th.matmul(A[idximag], B[idximag]), th.matmul(A[idxreal], B[idximag]) + th.matmul(A[idximag], B[idxreal])), dim=cdim), dim=dim, mode='matmul', dir='b')

def c2r(X, cdim=-1, keepdim=False):
    r"""complex representaion to real representaion

    Parameters
    ----------
    X : Tensor
        input in complex representaion
    cdim : int, optional
        real and imag dimension in real format, by default -1
    keepdim : bool, optional
        keep dimension, if :obj:`False`, stacks (make a new axis) at dimension :attr:`cdim`, 
        otherwise concatenates the real and imag part at exist dimension :attr:`cdim`, (Default is :obj:`False`).

    Returns
    -------
    tensor
        output in real representaion

    see also :func:`r2c`

    Examples
    ----------

    ::

        th.manual_seed(2020)
        Xr = th.randint(0, 30, (3, 3, 2))
        Xc = Xr[..., 0] + 1j * Xr[..., 1]
        Yr = c2r(Xc, cdim=0)
        Yc = r2c(Yr, cdim=0)
        print(Xr, Xr.shape, 'Xr')
        print(Xc, Xc.shape, 'Xc')
        print(Yr, Yr.shape, 'Yr')
        print(Yc, Yc.shape, 'Yc')

        # ---output
        tensor([[[20,  6],
                [27, 12],
                [25, 21]],

                [[21, 19],
                [29, 24],
                [25, 10]],

                [[16, 14],
                [ 6,  9],
                [ 5, 29]]]) torch.Size([3, 3, 2]) Xr
        tensor([[20.+6.j, 27.+12.j, 25.+21.j],
                [21.+19.j, 29.+24.j, 25.+10.j],
                [16.+14.j,  6.+9.j,  5.+29.j]]) torch.Size([3, 3]) Xc
        tensor([[[20., 27., 25.],
                [21., 29., 25.],
                [16.,  6.,  5.]],

                [[ 6., 12., 21.],
                [19., 24., 10.],
                [14.,  9., 29.]]]) torch.Size([2, 3, 3]) Yr
        tensor([[20.+6.j, 27.+12.j, 25.+21.j],
                [21.+19.j, 29.+24.j, 25.+10.j],
                [16.+14.j,  6.+9.j,  5.+29.j]]) torch.Size([3, 3]) Yc

    """
    if keepdim:
        return th.cat((X.real, X.imag), dim=cdim)
    else:
        return th.stack((X.real, X.imag), dim=cdim)


def r2c(X, cdim=-1, keepdim=False):
    r"""real representaion to complex representaion

    Parameters
    ----------
    X : Tensor
        input in real representaion
    cdim : int, optional
        real and imag dimension in real format, by default -1
    keepdim : bool, optional
        keep dimension, if :obj:`False`, discards axis :attr:`cdim`, 
        otherwise preserves the axis :attr:`cdim`, (Default is :obj:`False`). 
        (only work when the dimension at :attr:`cdim` equals 2)

    Returns
    -------
    tensor
        output in complex representaion

    see also :func:`c2r`

    Examples
    ----------

    ::

        th.manual_seed(2020)
        Xr = th.randint(0, 30, (3, 3, 2))
        Xc = Xr[..., 0] + 1j * Xr[..., 1]
        Yr = c2r(Xc, cdim=0)
        Yc = r2c(Yr, cdim=0)
        print(Xr, Xr.shape, 'Xr')
        print(Xc, Xc.shape, 'Xc')
        print(Yr, Yr.shape, 'Yr')
        print(Yc, Yc.shape, 'Yc')

        # ---output
        tensor([[[20,  6],
                [27, 12],
                [25, 21]],

                [[21, 19],
                [29, 24],
                [25, 10]],

                [[16, 14],
                [ 6,  9],
                [ 5, 29]]]) torch.Size([3, 3, 2]) Xr
        tensor([[20.+6.j, 27.+12.j, 25.+21.j],
                [21.+19.j, 29.+24.j, 25.+10.j],
                [16.+14.j,  6.+9.j,  5.+29.j]]) torch.Size([3, 3]) Xc
        tensor([[[20., 27., 25.],
                [21., 29., 25.],
                [16.,  6.,  5.]],

                [[ 6., 12., 21.],
                [19., 24., 10.],
                [14.,  9., 29.]]]) torch.Size([2, 3, 3]) Yr
        tensor([[20.+6.j, 27.+12.j, 25.+21.j],
                [21.+19.j, 29.+24.j, 25.+10.j],
                [16.+14.j,  6.+9.j,  5.+29.j]]) torch.Size([3, 3]) Yc

    """
    d, Nc = X.dim(), X.shape[cdim] // 2
    keepdim = keepdim if Nc == 1 else True
    if keepdim:
        return X[tb.sl(d, cdim, slice(0, Nc))] + 1j * X[tb.sl(d, cdim, slice(Nc, None))]
    else:
        return X[tb.sl(d, cdim, [0])] + 1j * X[tb.sl(d, cdim, [1])]


def conj(X, cdim=None):
    r"""conjugates a tensor

    Both complex and real representation are supported.

    Parameters
    ----------
    X : Tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued

    Returns
    -------
    tensor
         the inputs's conjugate matrix.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---conj')
        print(conj(X, cdim=0))
        print(conj(X[0] + 1j * X[1]))

        # ---output
        ---conj
        tensor([[[ 0.4869,  0.1052,  0.5883],
                [ 0.1161,  0.4949,  0.2824],
                [ 0.5899,  0.8105,  0.2512]],

                [[-0.6307, -0.5403, -0.8033],
                [-0.7781, -0.4966, -0.8888],
                [-0.5570, -0.7127, -0.0339]]])
        tensor([[0.4869-0.6307j, 0.1052-0.5403j, 0.5883-0.8033j],
                [0.1161-0.7781j, 0.4949-0.4966j, 0.2824-0.8888j],
                [0.5899-0.5570j, 0.8105-0.7127j, 0.2512-0.0339j]])

    """

    if th.is_complex(X):  # complex in complex
        return th.conj(X)
    else:
        if cdim is None:  # real
            return X
        else:  # complex in real
            d, Nc = X.dim(), X.shape[cdim] // 2
            return th.cat((X[tb.sl(d, axis=cdim, idx=slice(None, Nc))], -X[tb.sl(d, axis=cdim, idx=slice(Nc, None))]), dim=cdim)


def real(X, cdim=None, keepdim=False):
    r"""obtain real part of a tensor

    Both complex and real representation are supported.

    Parameters
    ----------
    X : Tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`) (only work when the dimension at :attr:`cdim` equals 2)

    Returns
    -------
    tensor
         the inputs's real part tensor.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---real')
        print(real(X, cdim=0))
        print(real(X[0] + 1j * X[1]))

        # ---output
        ---real
        tensor([[0.4869, 0.1052, 0.5883],
                [0.1161, 0.4949, 0.2824],
                [0.5899, 0.8105, 0.2512]])
        tensor([[0.4869, 0.1052, 0.5883],
                [0.1161, 0.4949, 0.2824],
                [0.5899, 0.8105, 0.2512]])
    """

    if th.is_complex(X):  # complex in complex
        return X.real
    else:
        if cdim is None:  # real
            return X
        else:  # complex in real
            d, Nc = X.ndim, X.shape[cdim] // 2
            keepdim = keepdim if Nc == 1 else True
            return X[tb.sl(d, axis=cdim, idx=slice(None, Nc))] if keepdim else X[tb.sl(d, axis=cdim, idx=0)]


def imag(X, cdim=None, keepdim=False):
    r"""obtain imaginary part of a tensor

    Both complex and real representation are supported.

    Parameters
    ----------
    X : Tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`) (only work when the dimension at :attr:`cdim` equals 2)

    Returns
    -------
    tensor
         the inputs's imaginary part tensor.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---imag')
        print(imag(X, cdim=0))
        print(imag(X[0] + 1j * X[1]))

        # ---output
        ---imag
        tensor([[0.6307, 0.5403, 0.8033],
                [0.7781, 0.4966, 0.8888],
                [0.5570, 0.7127, 0.0339]])
        tensor([[0.6307, 0.5403, 0.8033],
                [0.7781, 0.4966, 0.8888],
                [0.5570, 0.7127, 0.0339]])

    """

    if th.is_complex(X):  # complex in complex
        return X.imag
    else:
        if cdim is None:  # real
            return th.zeros_like(X)
        else:  # complex in real
            d, Nc = X.ndim, X.shape[cdim] // 2
            keepdim = keepdim if Nc == 1 else True
            return X[tb.sl(d, axis=cdim, idx=slice(Nc, None))] if keepdim else X[tb.sl(d, axis=cdim, idx=1)]

def angle(X, cdim=None, keepdim=False):
    r"""obtain angle of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm angle}(x) = {\rm atan}(\frac{v}{u}), x\in {\bf X}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : Tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`) (only work when the dimension at :attr:`cdim` equals 2)

    Returns
    -------
    tensor
         the inputs's amplitude.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---angle')
        print(angle(X))  # real
        print(angle(X, cdim=0))  # complex in real
        print(angle(X[0] + 1j * X[1]))  # complex in complex

    """

    if th.is_complex(X):  # complex in complex
        return X.angle()
    else:
        if cdim is None:  # real
            return X.angle()
        else:  # complex in real
            d, Nc = X.ndim, X.shape[cdim] // 2
            keepdim = keepdim if Nc == 1 else True
            idxreal = tb.sl(d, axis=cdim, idx=slice(None, Nc) if keepdim else [0])
            idximag = tb.sl(d, axis=cdim, idx=slice(Nc, None) if keepdim else [1])
            return th.arctan(X[idximag] / (X[idxreal] + tb.EPS))

def abs(X, cdim=None, keepdim=False):
    r"""obtain amplitude of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm abs}({\bf X}) = |x| = \sqrt{u^2 + v^2}, x\in {\bf X}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : Tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`) (only work when the dimension at :attr:`cdim` equals 2)

    Returns
    -------
    tensor
         the inputs's amplitude.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---abs')
        print(abs(X))  # real
        print(abs(X, cdim=0))  # complex in real
        print(abs(X[0] + 1j * X[1]))  # complex in complex

        # ---output
        ---abs
        tensor([[0.7968, 0.5504, 0.9957],
                [0.7868, 0.7011, 0.9326],
                [0.8113, 1.0793, 0.2535]])
        tensor([[0.7968, 0.5504, 0.9957],
                [0.7868, 0.7011, 0.9326],
                [0.8113, 1.0793, 0.2535]])
    """

    if th.is_complex(X):  # complex in complex
        return X.abs()
    else:
        if cdim is None:  # real
            return X.abs()
        else:  # complex in real
            d, Nc = X.ndim, X.shape[cdim] // 2
            keepdim = keepdim if Nc == 1 else True
            idxreal = tb.sl(d, axis=cdim, idx=slice(None, Nc) if keepdim else [0])
            idximag = tb.sl(d, axis=cdim, idx=slice(Nc, None) if keepdim else [1])
            return (X[idxreal]**2 + X[idximag]**2).sqrt()


def pow(X, cdim=None, keepdim=False):
    r"""obtain power of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm pow}({\bf X}) = |x|^2 = u^2 + v^2, x\in {\bf X}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : Tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`) (only work when the dimension at :attr:`cdim` equals 2)

    Returns
    -------
    tensor
         the inputs's power.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print('---pow')
        print(pow(X))  # real
        print(pow(X, cdim=0))  # complex in real
        print(pow(X[0] + 1j * X[1]))  # complex in complex

        # ---output
        ---pow
        tensor([[0.6349, 0.3030, 0.9914],
                [0.6190, 0.4915, 0.8697],
                [0.6583, 1.1649, 0.0643]])
        tensor([[0.6349, 0.3030, 0.9914],
                [0.6190, 0.4915, 0.8697],
                [0.6583, 1.1649, 0.0643]])
    """

    if th.is_complex(X):  # complex in complex
        return X.real*X.real + X.imag*X.imag
    else:
        if cdim is None:  # real
            return X**2
        else:  # complex in real
            d, Nc = X.ndim, X.shape[cdim] // 2
            keepdim = keepdim if Nc == 1 else True
            idxreal = tb.sl(d, axis=cdim, idx=slice(None, Nc) if keepdim else [0])
            idximag = tb.sl(d, axis=cdim, idx=slice(Nc, None) if keepdim else [1])
            return X[idxreal]**2 + X[idximag]**2

def mean(X, cdim=None, dim=None, keepdim=False):
    r"""mean

    Parameters
    ----------
    X : Tensor
        the input tensor
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int, list or None, optional
        the dimensions for calculation, by default None (all dims)
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    
    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print(mean(X))  # real
        print(mean(X, cdim=0))  # complex in real
        print(mean(X[0] + 1j * X[1]))  # complex in complex

    """    

    if th.is_complex(X):  # complex in complex
        return X.mean(dim=dim, keepdim=keepdim)
    else:
        if cdim is None:  # real
            return X.mean(dim=dim, keepdim=keepdim)
        else:  # complex in real
            dim = tb.rmcdim(X.ndim, dim=dim, cdim=cdim, keepdim=True)
            return X.mean(dim=dim, keepdim=keepdim)

def var(X, biased=False, cdim=None, dim=None, keepdim=False):
    r"""Calculates the variance over the specified dimensions

    .. math::
       \sigma^2=\frac{1}{N-\delta} \sum_{i=0}^{N-1}\left(x_i-\bar{x}\right)^2
    
    where :math:`\delta = 0` for biased estimation, :math:`\delta = 1` for unbiased estimation.
       
    Parameters
    ----------
    X : Tensor
        the input tensor
    biased : bool, optional
        :obj:`True` for N, :obj:`False` for N-1, by default :obj:`False`
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int, list or None, optional
        the dimensions for calculation, by default None (all dims)
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`)

    Returns
    -------
    tensor
        the result
            
    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print(var(X))  # real
        print(var(X, cdim=0))  # complex in real
        print(var(X[0] + 1j * X[1]))  # complex in complex

    """

    if th.is_complex(X):  # complex in complex
        return th.var(X, dim, correction=0 if biased else 1)
    else:
        if cdim is None:  # real
            return th.var(X, dim, correction=0 if biased else 1)
        else:  # complex in real
            if dim is None:
                N = X.numel() / 2
            elif type(dim) is int:
                N = X.shape[dim]
            else:
                N = 1
                for d in dim:
                    N *= X.shape[d]

            X = tb.pow(X - mean(X, cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True).sum(dim=dim, keepdim=keepdim)

            if biased:
                return X / N
            else:
                return X / (N - 1)

def std(X, biased=False, cdim=None, dim=None, keepdim=False):
    r"""Calculates the standard deviation over the specified dimensions

    Parameters
    ----------
    X : Tensor
        the input tensor
    biased : bool, optional
        :obj:`True` for N, :obj:`False` for N-1, by default :obj:`False`
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int, list or None, optional
        the dimensions for calculation, by default None (all dims)
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`)

    Returns
    -------
    tensor
        the result

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print(std(X))  # real
        print(std(X, cdim=0))  # complex in real
        print(std(X[0] + 1j * X[1]))  # complex in complex

    """

    if th.is_complex(X):  # complex in complex
        return th.std(X, dim, correction=0 if biased else 1, keepdim=keepdim)
    else:
        if cdim is None:  # real
            return th.std(X, dim, correction=0 if biased else 1, keepdim=keepdim)
        else:  # complex in real
            if dim is None:
                N = X.numel() / 2
            elif type(dim) is int:
                N = X.shape[dim]
            else:
                N = 1
                for d in dim:
                    N *= X.shape[d]

            X = pow(X - mean(X, cdim=cdim, dim=dim, keepdim=True), cdim=cdim, keepdim=True).sum(dim=dim, keepdim=keepdim)

            if biased:
                return (X / N).sqrt()
            else:
                return (X / (N - 1)).sqrt()

def cov(X, Y, biased=False, cdim=None, dim=None, keepdim=False):
    r"""Calculates the covariance over the specified dimensions

    .. math::
       \operatorname{cov}_w(x, y)=\frac{\sum_{i=1}^N\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)}{N-\delta}

    where :math:`\delta = 0` for biased estimation, :math:`\delta = 1` for unbiased estimation.

    Parameters
    ----------
    X : Tensor
        the first input tensor
    Y : Tensor
        the second input tensor
    biased : bool, optional
        :obj:`True` for N, :obj:`False` for N-1, by default :obj:`False`
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int, list or None, optional
        the dimensions for calculation, by default None (all dims)
    keepdim : bool, optional
        keep dimensions? (include complex dim, defalut is :obj:`False`)

    Returns
    -------
    tensor
        the result
            
    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))
        Y = th.rand((2, 3, 3))

        print(cov(X, Y))  # real
        print(cov(X, Y, cdim=0))  # complex in real
        print(cov(X[0] + 1j * X[1], Y[0] + 1j * Y[1]))  # complex in complex

    """

    X = X - tb.mean(X, cdim=cdim, dim=dim, keepdim=True)
    Y = Y - tb.mean(Y, cdim=cdim, dim=dim, keepdim=True)

    C = tb.dot(X, Y, mode='xyh', cdim=cdim, dim=dim, keepdim=keepdim)
    dim = list(range(X.ndim)) if dim is None else [dim] if type(dim) is int else dim

    if cdim is None:
        N = X.numel() if dim is None else np.prod([X.shape[d] for d in dim])
    else:
        N = X.numel() // 2 if dim is None else np.prod([X.shape[d] for d in dim])

    if biased:
        C = C / N
    else:
        C = C / (N - 1)

    return C

def dot(X, Y, mode='xyh', cdim=None, dim=None, keepdim=False):
    r"""dot product or inner product

    .. math::
       <x,y> = xy^H

    .. note:: 
       the :func:`dot` function in numpy and pytorch compute the inner product by :math:`<x,y> = xy`.

    Parameters
    ----------
    X : Tensor
        the left input
    Y : Tensor
        the right input
    mode : str
        ``'xyh'`` for :math:`<x,y> = xy^H` (default), ``'xy'`` for :math:`<x,y> = xy`, where :math:`y^H` is the complex conjection of :math:`y`
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : tuple, None, optional
        The dimension axis for computing dot product. The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.rand((2, 3, 3))

        print(dot(X, X))  # real
        print(dot(X, X, cdim=0))  # complex in real
        print(dot(X[0] + 1j * X[1], X[0] + 1j * X[1]))  # complex in complex

    """

    dim = tb.rmcdim(X.ndim, dim=dim, cdim=cdim, keepdim=True)
    if mode.lower() == 'xyh':
        return ematmul(X, tb.conj(Y, cdim=cdim), cdim=cdim).sum(dim=dim, keepdim=keepdim)
    elif mode.lower() == 'xy':
        return ematmul(X, Y, cdim=cdim).sum(dim=dim, keepdim=keepdim)
    else:
        raise ValueError('mode %s is not support!' % mode)


if __name__ == '__main__':

    print(prevpow2(-5), nextpow2(-5))
    print(prevpow2(5), nextpow2(5))
    print(prevpow2(0.3), nextpow2(0.3))
    print(prevpow2(7.3), nextpow2(7.3))
    print(prevpow2(-3.5), nextpow2(-3.5))

    th.manual_seed(2020)
    Ar = th.randn((3, 3, 2))
    Br = th.randn((3, 3, 2))

    Ac = th.view_as_complex(Ar)
    Bc = th.view_as_complex(Br)

    Mr = th.view_as_real(Ac * Bc)
    print(th.sum(Mr - ematmul(Ar, Br, cdim=-1)))
    print(th.sum(Ac * Bc - ematmul(Ac, Bc)))

    print(th.sum(th.matmul(Ac, Bc) - matmul(Ac, Bc)))
    Mr = matmul(Ar, Br, cdim=-1, dim=( 0, 1))
    Mc = th.view_as_real(th.matmul(Ac, Bc))
    print(th.sum(Mr - Mc))

    th.manual_seed(2020)
    Xr = th.randint(0, 30, (3, 3, 2))
    Xc = Xr[..., 0] + 1j * Xr[..., 1]
    Yr = c2r(Xc, cdim=0)
    Yc = r2c(Yr, cdim=0)
    print(Xr, Xr.shape, 'Xr')
    print(Xc, Xc.shape, 'Xc')
    print(Yr, Yr.shape, 'Yr')
    print(Yc, Yc.shape, 'Yc')

    th.manual_seed(2020)
    X = th.rand((2, 3, 3))

    print('---conj')
    print(conj(X, cdim=0))
    print(conj(X[0] + 1j * X[1]))

    print('---real')
    print(real(X, cdim=0))
    print(real(X[0] + 1j * X[1]))

    print('---imag')
    print(imag(X, cdim=0))
    print(imag(X[0] + 1j * X[1]))

    print('---abs')
    print(abs(X, cdim=0))
    print(abs(X[0] + 1j * X[1]))

    print('---pow')
    print(pow(X, cdim=0))
    print(pow(X[0] + 1j * X[1]))

    print('---mean')
    print(mean(X, cdim=0))
    print(mean(X[0] + 1j * X[1]))

    print('---var')
    print(var(X, cdim=0))
    print(var(X[0] + 1j * X[1]))

    print('---std')
    print(std(X, cdim=0))
    print(std(X[0] + 1j * X[1]))

    print('---cov')
    print(cov(X, X))
    print(cov(X, X, cdim=0))
    print(cov(X[0] + 1j * X[1], X[0] + 1j * X[1]))
    print('---cov')
    print(cov(X, X, dim=(-2, -1)))
    print(cov(X, X, cdim=0, dim=(-2, -1)))
    print(cov(X[0] + 1j * X[1], X[0] + 1j * X[1], dim=(-2, -1)))

    print('---dot')
    print(dot(X, X))
    print(dot(X, X, cdim=0))
    print(dot(X[0] + 1j * X[1], X[0] + 1j * X[1]))
    print('---dot')
    print(dot(X, X, dim=(-2, -1)))
    print(dot(X, X, cdim=0, dim=(-2, -1)))
    print(dot(X[0] + 1j * X[1], X[0] + 1j * X[1], dim=(-2, -1)))

    print('---fnab')
    print(fnab(5))
    print(fnab(6))
    print(fnab(7))
    print(fnab(8))
    print(fnab(9))

    print('---db2mag')
    print(db2mag(30))
    print(db2mag([30]))
    print(db2mag(np.array([30])))
    print(db2mag(th.tensor([30])))
    print('---mag2db')
    print(mag2db(31.62277))
    print(mag2db([31.62277]))
    print(mag2db(np.array([31.62277])))
    print(mag2db(th.tensor([31.62277])))
