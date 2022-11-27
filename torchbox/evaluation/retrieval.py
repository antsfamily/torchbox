#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : retrieval.py
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


def true_positive(X, Y):
    """Find true positive elements

    true_positive(X, Y) returns those elements that are positive classes in Y
    and retrieved as positive in X.

    Parameters
    ----------
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
        referenced, positive-->1, negative-->0

    Returns
    -------
    TP: tensor
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In TP, true positive elements are ones, while others are zeros.
    """
    TP = ((X == 1) + (Y == 1)) == 2
    return TP


def false_positive(X, Y):
    """Find false positive elements

    false_positive(X, Y) returns elements that are negative classes in Y
    and retrieved as positive in X.

    Parameters
    ----------
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
        referenced, positive-->1, negative-->0

    Returns
    -------
    FP: tensor
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In FP, false positive elements are ones, while others are zeros.
    """
    FP = ((X == 1) + (Y == 0)) == 2
    return FP


def true_negative(X, Y):
    """Find true negative elements

    true_negative(X, Y) returns elements that are negative classes in Y
    and retrieved as negative in X.

    Parameters
    ----------
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
        referenced, positive-->1, negative-->0

    Returns
    -------
    TN: tensor
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In TN, true negative elements are ones, while others are zeros.
    """
    TN = ((X == 0) + (Y == 0)) == 2
    return TN


def false_negative(X, Y):
    """Find false negative elements

    true_negative(X, Y) returns elements that are positive classes in Y
    and retrieved as negative in X.

    Parameters
    ----------
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
        referenced, positive-->1, negative-->0

    Returns
    -------
    FN: tensor
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In FN, false negative elements are ones, while others are zeros.
    """
    FN = ((X == 0) + (Y == 1)) == 2
    return FN


def precision(X, Y, TH=None):
    r"""Compute precision

    .. math::
       {\rm PPV} = {P} = \frac{\rm TP}{{\rm TP} + {\rm FP}}
       :label: equ-Precision

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
    P: float
        precision
    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    # TP : True Positive
    # FP : False Positive
    TP = true_positive(X, Y)
    FP = false_positive(X, Y)
    P = float(th.sum(TP)) / (float(th.sum(TP + FP)) + EPS)
    return P


def recall(X, Y, TH=None):
    r"""Compute recall(sensitivity)

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

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
    R: float
        recall
    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    # TP : True Positive
    # FN : False Negative
    TP = true_positive(X, Y)
    FN = false_negative(X, Y)
    R = float(th.sum(TP)) / (float(th.sum(TP + FN)) + EPS)
    return R


def sensitivity(X, Y, TH=None):
    r"""Compute sensitivity(recall)

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

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
    R: float
        recall
    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    # TP : True Positive
    # FN : False Negative
    TP = true_positive(X, Y)
    FN = false_negative(X, Y)
    R = float(th.sum(TP)) / (float(th.sum(TP + FN)) + EPS)
    return R


def selectivity(X, Y, TH=None):
    r"""Compute selectivity or specificity

    .. math::
       {\rm TNR} = {S} = \frac{\rm TN}{{\rm TN} + {\rm FP}}
       :label: equ-selectivity

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
    S: float
        selectivity
    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    # TN : True Negative
    # FP : False Positive
    TN = true_negative(X, Y)
    FP = false_positive(X, Y)
    S = float(th.sum(TN)) / (float(th.sum(TN + FP)) + EPS)
    return S


def fmeasure(X, Y, TH=None, beta=1.0):
    r"""Compute F-measure

    .. math::
       F_{\beta} = \frac{(1+\beta^2)PR}{\beta^2P + R}
       :label: equ-F-measure

    Parameters
    ----------
    X : tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0
    beta : float
        X > TH --> 1, X <= TH --> 0
    Returns
    -------
    F: float
        F-measure
    """
    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    TP = true_positive(X, Y)
    FP = false_positive(X, Y)
    P = float(th.sum(TP)) / (float(th.sum(TP + FP)) + EPS)

    TP = true_positive(X, Y)
    FN = false_negative(X, Y)
    R = float(th.sum(TP)) / (float(th.sum(TP + FN)) + EPS)
    F = (1 + beta**2) * P * R / ((beta**2) * P + R + EPS)
    return F


def false_alarm_rate(X, Y, TH=None):
    r"""Compute false alarm rate or False Discovery Rate

    .. math::
       {\rm FDR} = \frac{\rm FP}{{\rm TP} + {\rm FP}} = 1 - P
       :label: equ-FalseDiscoveryRate

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
    FDR: float
        False Discovery Rate
    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    # FP : False Positive
    # TP : True Positive
    TP = true_positive(X, Y)
    FP = false_positive(X, Y)
    FDR = float(th.sum(FP)) / (float(th.sum(TP + FP)) + EPS)
    return FDR


def miss_alarm_rate(X, Y, TH=None):
    r"""Compute miss alarm rate or False Negative Rate

    .. math::
       {\rm FNR} = \frac{\rm FN}{{\rm FN} + {\rm TP}} = 1 - R
       :label: equ-FalseNegativeRate

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
    FNR: float
        False Negative Rate
    """

    if TH is not None:
        X = (X > TH).float()

    X = (X > 0.5)
    Y = (Y > 0.5)

    # FP : False Negative
    # TP : True Positive
    TP = true_positive(X, Y)
    FN = false_negative(X, Y)
    FNR = float(th.sum(FN)) / (float(th.sum(TP + FN)) + EPS)
    return FNR


if __name__ == '__main__':
    import numpy as np
    import torchbox as tb

    X = np.array([[1, 1, 1], [0, 1, 0]])
    X = np.array([[0, 0, 1], [0, 0, 1]])
    X = np.array([[0.2, 0, 1], [0, 0.2, 1]])
    Y = np.array([[0, 0, 1], [0, 0, 1]])
    TH = 0.1
    beta = 1.0

    X = th.Tensor(X)
    Y = th.Tensor(Y)
    prec = tb.precision(X, Y)
    print("prec: ", prec)
    acc = tb.accuracy(X, Y)
    print("acc: ", acc)

    P = precision(X, Y, TH=TH)
    R = recall(X, Y, TH=TH)
    F = fmeasure(X, Y, TH=TH, beta=1.0)
    FDR = false_alarm_rate(X, Y, TH=None)
    FNR = miss_alarm_rate(X, Y, TH=None)

    print("P: ", P)
    print("R: ", R)
    print("F: ", F)
    print("FDR: ", FDR)
    print("FNR: ", FNR)
    print("1.0-FDR: ", 1.0 - FDR)
    print("1.0-FNR: ", 1.0 - FNR)

    N, H, W = (10, 3, 4)
    Xs = th.zeros(N, H, W)
    Ys = th.zeros(N, H, W)

    Xs[5, 1, 2] = 1
    Ys[5, 1, 2] = 1

    Fall = fmeasure(Xs, Ys, TH=TH, beta=1.0)

    Favg = 0.
    for X, Y in zip(Xs, Ys):
        # print(th.sum(X), th.sum(Y))
        Favg += fmeasure(X, Y, beta=1.0)
        # print(Favg)

    Favg = Favg / N

    print("Fall, Favg: ", Fall, Favg)
