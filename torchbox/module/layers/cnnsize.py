#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : cnnsize.py
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

import numpy as np


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
       :label: equ-TransposeConv2dSize

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
