#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : window_function.py
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


def window(n, wtype=None, isperiodic=True, dtype=None, device=None, requires_grad=False):
    r"""Generates window

    Parameters
    ----------
    n : int
        The length of the window.
    wtype : str or None, optional
        The window type:
        - ``'rectangle'`` for rectangle window
        - ``'bartlett'`` for bartlett window
        - ``'blackman'`` for blackman window
        - ``'hamming x y'`` for hamming window with :math:`\alpha=x, \beta=y`, default is 0.54, 0.46.
        - ``'hanning'`` for hanning window
        - ``'kaiser x'`` for kaiser window with :math:`\beta=x`, default is 12.
    isperiodic : bool, optional
        If True (default), returns a window to be used as periodic function.
        If False, return a symmetric window.
    dtype : None, optional
        The desired data type of returned tensor.
    device : None, optional
        The desired device of returned tensor.
    requires_grad : bool, optional
        If autograd should record operations on the returned tensor. Default: False.

    Returns
    -------
    tensor
        A 1-D tensor of size (n,) containing the window
    """

    if wtype is None:
        return th.ones(n, dtype=dtype, device=device, requires_grad=requires_grad)

    wtype.lower()
    wtype = wtype.split()

    if wtype[0] == 'rectangle':
        return th.ones(n, dtype=dtype, device=device, requires_grad=requires_grad)
    if wtype[0] == 'bartlett':
        return th.bartlett_window(n, periodic=isperiodic, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)
    if wtype[0] == 'blackman':
        return th.blackman_window(n, periodic=isperiodic, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)
    if wtype[0] == 'hamming':
        alpha, beta = 0.54, 0.46
        if len(wtype) == 2:
            alpha = float(wtype[1])
        elif len(wtype) == 3:
            alpha, beta = float(wtype[1]), float(wtype[2])
        return th.hamming_window(n, periodic=isperiodic, alpha=alpha, beta=beta, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)
    if wtype[0] == 'hanning':
        return th.hann_window(n, periodic=isperiodic, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)
    if wtype[0] == 'kaiser':
        beta = 12.
        if len(wtype) > 1:
            beta = float(wtype[1])
        return th.kaiser_window(n, periodic=isperiodic, beta=beta, dtype=dtype, layout=th.strided, device=device, requires_grad=requires_grad)


def windowing(x, w, axis=None):
    """Performs windowing operation in the specified axis.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    w : Tensor
        A 1-d window tensor.
    axis : int or None, optional
        The axis.

    Returns
    -------
    tensor
        The windowed data.

    """
    if axis is None:
        return x * w

    if type(axis) is not int:
        raise TypeError('The axis should be a integer!')

    d = x.dim()
    shape = [1] * d
    shape[axis] = len(w)

    w = w.view(shape)
    return x * w


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 512
    wtype = 'bartlett'
    wtype = 'blackman'
    wtype = 'hamming 0.54 0.46'
    wtype = 'hanning'
    wtype = 'kaiser 12'
    w = window(n, wtype=wtype)

    plt.figure()
    plt.grid()
    plt.plot(w)
    plt.show()
