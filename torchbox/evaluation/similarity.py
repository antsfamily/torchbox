#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : similarity.py
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


def jaccard_index(X, Y, TH=None):
    r"""Jaccard similarity coefficient

    .. math::
        \mathrm{J}(\mathrm{A}, \mathrm{B})=\frac{|A \cap B|}{|A \cup B|}

    Parameters
    ----------
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
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
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
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
