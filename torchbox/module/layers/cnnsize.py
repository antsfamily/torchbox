#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : cnnsize.py
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

import numpy as np


def conv_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
    r"""computes output shape of convolution

    .. math::
       \begin{array}{l}
       H_{o} &= \left\lfloor\frac{H_{i}  + 2 \times P_h - D_h \times (K_h - 1) - 1}{S_h} + 1\right\rfloor \\
       W_{o} &= \left\lfloor\frac{W_{i}  + 2 \times P_w - D_w \times (K_w - 1) - 1}{S_w} + 1\right\rfloor \\
       B_{o} &= \left\lfloor\frac{B_{i}  + 2 \times P_b - D_w \times (K_b - 1) - 1}{S_b} + 1\right\rfloor \\
        \cdots
       \end{array}
       :label: equ-DilationConvxdSize

    Parameters
    ----------
    in_size : list or tuple
        the size of input (without batch and channel)
    kernel_size : int, list or tuple
        the window size of convolution
    stride : int, list or tuple, optional
        the stride of convolution, by default 1
    padding : int, str, list or tuple, optional
        the padding size of convolution, ``'valid'``, ``'same'``, by default 0
    dilation : int, list or tuple, optional
        the spacing between kernel elements, by default 1
    """

    ndim = len(in_size)
    if padding in ['same', 'SAME']:
        return in_size.copy()

    if padding in ['valid', 'VALID']:
        padding = [0]*ndim

    kernel_size = [kernel_size] * ndim if type(kernel_size) is int else kernel_size * ndim if len(kernel_size) == 1 else kernel_size
    dilation = [dilation] * ndim if type(dilation) is int else dilation * ndim if len(dilation) == 1 else dilation
    stride = [stride] * ndim if type(stride) is int else stride * ndim if len(stride) == 1 else stride
    out_size = []
    for n in range(ndim):
        out_size.append(int(np.floor((in_size[n] + 2 * padding[n] - dilation[n] * (kernel_size[n] - 1) - 1) / stride[n] + 1)))
    return out_size

def ConvSize1d(CLi, Co, K, S, P, D=1, groups=1):
    r"""Compute shape after 2D-Convolution

    .. math::
       \begin{array}{l}
       L_{o} &= \left\lfloor\frac{L_{i}  + 2 \times P_l - D_l \times (K_l - 1) - 1}{S_l} + 1\right\rfloor \\
       \end{array}

    CLi : tuple or list
        input data shape (C, L)
    Co : int
        number of output chanels.
    K : tuple
        kernel size
    S : tuple
        stride size
    P : tuple
        padding size
    D : tuple, optional
        dilation size (the default is 1)
    groups : int, optional
        1 (the default is 1)

    Returns
    -------
    tuple
        shape after 2D-Convolution

    Raises
    ------
    ValueError
        dilation should be greater than zero.
    """
    if D < 1:
        raise ValueError("dilation should be greater than zero")
    Ci, Li = CLi

    Lo = int(np.floor((Li + 2 * P - D * (K - 1) - 1) / S + 1))

    return Co, Lo


def ConvTransposeSize1d(CLi, Co, K, S, P, D=1, OP=0, groups=1):
    r"""Compute shape after Transpose Convolution

    .. math::
       \begin{array}{l}
       L_{o} &= (L_{i} - 1) \times S_l - 2 \times P_l + D_l \times (K_l - 1) + OP_l + 1 \\
       \end{array}
       :label: equ-TransposeConv1dSize

    Parameters
    ----------
    CLi : tuple or list
        input data shape (C, H, W)
    Co : int
        number of output chanels.
    K : tuple
        kernel size
    S : tuple
        stride size
    P : tuple
        padding size
    D : tuple, optional
        dilation size (the default is 1)
    OP : tuple, optional
        output padding size (the default is 0)
    groups : int, optional
        one group (the default is 1)

    Returns
    -------
    tuple
        shape after 2D-Transpose Convolution

    Raises
    ------
    ValueError
        output padding must be smaller than either stride or dilation
    """

    if not ((OP < S) or (OP < D)):
        raise ValueError(
            "output padding must be smaller than either stride or dilation")
    Ci, Li = CLi

    Lo = int((Li - 1) * S - 2 * P + D * (K - 1) + OP + 1)

    return Co, Lo


def PoolSize1d(CLi, K, S, P, D=1):
    Ci, Li = CLi

    Lo = int(np.floor((Li + 2 * P - D * (K - 1) - 1) / S + 1))

    return Ci, Lo


def UnPoolSize1d(CLi, K, S, P, D=1):
    Ci, Li = CLi

    Lo = int((Li - 1) * S - 2 * P + K)

    return Ci, Lo


def ConvSize2d(CHWi, Co, K, S, P, D=(1, 1), groups=1):
    r"""Compute shape after 2D-Convolution

    .. math::
       \begin{array}{l}
       H_{o} &= \left\lfloor\frac{H_{i}  + 2 \times P_h - D_h \times (K_h - 1) - 1}{S_h} + 1\right\rfloor \\
       W_{o} &= \left\lfloor\frac{W_{i}  + 2 \times P_w - D_w \times (K_w - 1) - 1}{S_w} + 1\right\rfloor
       \end{array}
       :label: equ-DilationConv2dSize

    CHWi : tuple or list
        input data shape (C, H, W)
    Co : int
        number of output chanels.
    K : tuple
        kernel size
    S : tuple
        stride size
    P : tuple
        padding size
    D : tuple, optional
        dilation size (the default is (1, 1))
    groups : int, optional
        [description] (the default is 1, which [default_description])

    Returns
    -------
    tuple
        shape after 2D-Convolution

    Raises
    ------
    ValueError
        dilation should be greater than zero.
    """
    if D[0] + D[1] < 2:
        raise ValueError("dilation should be greater than zero")
    Ci, Hi, Wi = CHWi

    Ho = int(np.floor((Hi + 2 * P[0] - D[0] * (K[0] - 1) - 1) / S[0] + 1))
    Wo = int(np.floor((Wi + 2 * P[1] - D[1] * (K[1] - 1) - 1) / S[1] + 1))

    return Co, Ho, Wo


def ConvTransposeSize2d(CHWi, Co, K, S, P, D=(1, 1), OP=(0, 0), groups=1):
    r"""Compute shape after Transpose Convolution

    .. math::
       \begin{array}{l}
       H_{o} &= (H_{i} - 1) \times S_h - 2 \times P_h + D_h \times (K_h - 1) + OP_h + 1 \\
       W_{o} &= (W_{i} - 1) \times S_w - 2 \times P_w + D_w \times (K_w - 1) + OP_w + 1
       \end{array}
       :label: equ-TransposeConv2dSize

    Parameters
    ----------
    CHWi : tuple or list
        input data shape (C, H, W)
    Co : int
        number of output chanels.
    K : tuple
        kernel size
    S : tuple
        stride size
    P : tuple
        padding size
    D : tuple, optional
        dilation size (the default is (1, 1))
    OP : tuple, optional
        output padding size (the default is (0, 0))
    groups : int, optional
        one group (the default is 1)

    Returns
    -------
    tuple
        shape after 2D-Transpose Convolution

    Raises
    ------
    ValueError
        output padding must be smaller than either stride or dilation
    """

    if not ((OP[0] < S[0] and OP[1] < S[1]) or (OP[0] < D[0] and OP[1] < D[1])):
        raise ValueError(
            "output padding must be smaller than either stride or dilation")
    Ci, Hi, Wi = CHWi

    Ho = int((Hi - 1) * S[0] - 2 * P[0] + D[0] * (K[0] - 1) + OP[0] + 1)
    Wo = int((Wi - 1) * S[1] - 2 * P[1] + D[1] * (K[1] - 1) + OP[1] + 1)

    return Co, Ho, Wo


def PoolSize2d(CHWi, K, S, P, D=(1, 1)):
    Ci, Hi, Wi = CHWi

    Ho = int(np.floor((Hi + 2 * P[0] - D[0] * (K[0] - 1) - 1) / S[0] + 1))
    Wo = int(np.floor((Wi + 2 * P[1] - D[1] * (K[1] - 1) - 1) / S[1] + 1))

    return Ci, Ho, Wo


def UnPoolSize2d(CHWi, K, S, P, D=(1, 1)):
    Ci, Hi, Wi = CHWi

    Ho = int((Hi - 1) * S[0] - 2 * P[0] + K[0])
    Wo = int((Wi - 1) * S[1] - 2 * P[1] + K[1])

    return Ci, Ho, Wo


if __name__ == '__main__':
    import torchbox as tb
    import torch as th

    n = 2
    CHWi = (4, 12, 12)
    Co = 16
    K = (3, 3)
    S = (2, 2)
    P = (1, 1)
    OP = (1, 1)
    D = (1, 1)

    print('===Conv2d')
    print(CHWi)
    print('---Theoretical result')

    CHWo = tb.ConvSize2d(CHWi=CHWi, Co=Co, K=K,
                          S=S, P=P, D=D)
    print(CHWo)

    print(conv_size(in_size=(12, 12), kernel_size=K, stride=S, padding=P, dilation=D))

    print('---Torch result')
    x = th.randn((n, ) + CHWi)
    print(x.size())
    conv = th.nn.Conv2d(CHWi[0], Co, kernel_size=K,
                        stride=S, padding=P, dilation=D)
    y = conv(x)
    print(y.size())

    print('===Deconv2d')

    CHWo = tb.ConvTransposeSize2d(CHWi=CHWo, Co=CHWi[0], K=K,
                                   S=S, P=P, D=D, OP=OP)
    print('---Theoretical result')
    print(CHWo)

    print('---Torch result')

    upconv = th.nn.ConvTranspose2d(Co, CHWi[0], kernel_size=K,
                                   stride=S, padding=P, output_padding=OP, dilation=D)
    xx = upconv(y)
    print(xx.size())

    print("===Pool2d")
    print('---Theoretical result')
    CHWo = tb.PoolSize2d(CHWi, K, S, P, D=D)
    print(CHWo)
    print('---Torch result')
    pool = th.nn.MaxPool2d(kernel_size=K, stride=S,
                           padding=P, dilation=D, return_indices=True)
    y, idx = pool(x)
    print(y.size())
    # print(idx)
    # print(y)

    print("===UnPool2d")
    print('---Theoretical result')
    CHWo = tb.UnPoolSize2d(CHWo, K, S, P, D=D)
    print(CHWo)
    print('---Torch result')
    unpool = th.nn.MaxUnpool2d(kernel_size=K, stride=S, padding=P)
    xx = unpool(y, idx, output_size=x.size())
    print(xx.size())
