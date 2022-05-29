#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

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
