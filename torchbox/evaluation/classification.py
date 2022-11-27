#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : classification.py
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


def accuracy(X, Y, TH=None):
    r"""compute accuracy



    Parameters
    ----------
    X : tensor
        Predicted one hot matrix, :math:`\{0, 1\}`
    Y : tensor
        Referenced one hot matrix, :math:`\{0, 1\}`
    TH : float, optional
        threshold: X > TH --> 1, X <= TH --> 0
    """

    if TH is not None:
        X = (X > TH).float()

    acc = th.mean((X == Y).float()).item()

    return acc


if __name__ == '__main__':
    import numpy as np
    import torchbox as tb

    P = np.array([[1, 1, 1], [0, 1, 0]])
    R = np.array([[0, 0, 1], [0, 0, 1]])

    P = th.Tensor(P)
    R = th.Tensor(R)
    prec = tb.precision(P, R)
    print(prec)
    acc = accuracy(P, R)
    print(acc)
