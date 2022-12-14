#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : retrieval.py
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
from torch.autograd import Variable


class DiceLoss(th.nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def soft_dice_coeff(self, P, G):

        cupnum = th.sum(P, (1, 2, 3)) + th.sum(G, (1, 2, 3))
        capnum = th.sum(P * G, (1, 2, 3))

        score = (2. * capnum + EPS) / (cupnum + EPS)
        if self.reduce is True:
            if self.size_average is True:
                score = score.mean()
            else:
                score = score.sum()

        return score

    def __call__(self, P, G):
        return 1. - self.soft_dice_coeff(P, G)


class JaccardLoss(th.nn.Module):
    r"""Jaccard distance

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

    def __init__(self, size_average=True, reduce=True):
        super(JaccardLoss, self).__init__()

    def forward(self, P, G):
        capnum = th.sum(P * G)
        cupnum = th.sum(P + G)

        return (1.0 - capnum / (cupnum - capnum + EPS))


class IridescentLoss(th.nn.Module):
    r"""Iridescent Distance Loss

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

    def __init__(self, size_average=True, reduce=True):
        super(IridescentLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, P, G):
        sumbothones = th.sum(P * G)
        sumbothzeros = th.sum((1 - P) * (1 - G))
        print(sumbothones, sumbothzeros)
        return 1.0 - (sumbothones + sumbothzeros) / (th.sum(P) + th.sum(G) - sumbothones + sumbothzeros + EPS)
        # return 1.0 - (th.sum(P * G) + th.sum((1 - P) * (1 - G))) / (th.sum(P) + th.sum(G) - th.sum(P * G) + EPS)


class F1Loss(th.nn.Module):
    r"""F1 distance Loss

    .. math::
       F_{\beta} = 1 -\frac{(1+\beta^2) * P * R}{\beta^2 *P + R}

    where,

    .. math::
       {\rm PPV} = {P} = \frac{\rm TP}{{\rm TP} + {\rm FP}}

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}

    """

    def __init__(self, size_average=True, reduce=True):
        super(F1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, P, G):

        TP = th.sum(P * G)
        FP = th.sum(P * (1. - G))
        FN = th.sum((1 - P) * G)
        Precision = TP / (TP + FP + EPS)
        Recall = TP / (TP + FN + EPS)

        xx = 2.0 * Precision * Recall / (Precision + Recall + EPS)
        return 1.0 - xx
        # return 1.0 / (2.0 * P * R / (P + R + EPS) + EPS)


if __name__ == '__main__':
    import numpy as np

    # X = np.array([[1, 1, 1], [0, 1, 0]])
    # X = np.array([[0, 0, 1], [0, 0, 1]])
    X = np.array([[0.3, 0, 1], [0, 0.8, 1]])
    X = np.array([[0, 0, 1], [0, 0, 1]])
    G = np.array([[0, 0, 1], [0, 0, 1]])

    X = np.array([[0.1, 0.1, 0], [0, 0, 0]])
    G = np.array([[0, 0.5, 0], [0, 0, 0]])

    # X = np.array([[1, 1, 0], [0, 0, 0]])
    # G = np.array([[1, 1, 1], [1, 1, 1]])

    # X = np.array([[1, 1, 1], [1, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 0, 0]])

    # X = np.array([[0, 1, 0], [0, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 1, 0]])

    # X = np.array([[0, 0, 0], [0, 0, 0]])
    # G = np.array([[0, 0, 0], [0, 0, 0]])

    # X = np.array([[1, 0, 1], [1, 1, 1]])
    # G = np.array([[1, 1, 1], [1, 1, 1]])

    X = np.array([[1, 1, 1], [1, 1, 1]])
    G = np.array([[1, 1, 1], [1, 1, 1]])

    X = th.randn(4, 1, 3, 2)
    G = th.randn(4, 1, 3, 2)

    X = th.Tensor(X)
    G = th.Tensor(G)
    X = Variable(X, requires_grad=True)
    G = Variable(G, requires_grad=True)

    net = th.nn.ReLU()
    P = net(X)

    print(X)
    print(P)
    print(G)
    print(th.mean(th.abs(X - G)))
