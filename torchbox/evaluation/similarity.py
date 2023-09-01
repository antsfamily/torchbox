#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : similarity.py
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
from torchbox.utils.const import EPS


def jaccard_index(X, Y, TH=None):
    r"""Jaccard similarity coefficient

    .. math::
        \mathrm{J}(\mathrm{A}, \mathrm{B})=\frac{|A \cap B|}{|A \cup B|}

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    JS : float
        the jaccard similarity coefficient.

    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    X = th.as_tensor(X, dtype=th.float32)
    Y = th.as_tensor(Y, dtype=th.float32)

    Inter = th.sum((X + Y) == 2)
    Union = th.sum((X + Y) >= 1)

    JS = float(Inter) / (float(Union) + EPS)

    return JS


def dice_coeff(X, Y, TH=0.5):
    r"""Dice coefficient

    .. math::
        s = \frac{2|Y \cap X|}{|X|+|Y|}

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    DC : float
        the dice coefficient.
    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    X = th.as_tensor(X, dtype=th.float32)
    Y = th.as_tensor(Y, dtype=th.float32)
    Inter = th.sum((X + Y) == 2)
    DC = float(2 * Inter) / (float(th.sum(X) + th.sum(Y)) + EPS)

    return DC


if __name__ == '__main__':
    import numpy as np
    import torchbox as tb

    X = np.array([[1, 1, 1], [0, 1, 0]])
    Y = np.array([[0, 0, 1], [0, 0, 1]])
    TH = None

    X = th.Tensor(X)
    Y = th.Tensor(Y)
    prec = tb.precision(X, Y)
    print("prec: ", prec)
    acc = tb.accuracy(X, Y)
    print("acc: ", acc)

    JS = jaccard_index(X, Y, TH=TH)
    DC = dice_coeff(X, Y, TH=TH)

    print("JS: ", JS)
    print("DC: ", DC)
    print("2JS/(1+JS)", 2.0 * JS / (1.0 + JS))
    print("DC/(2-DC)", DC / (2.0 - DC))
