#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : polynomialfit.py
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
from torch.nn.parameter import Parameter


class PolyFit(th.nn.Module):
    r"""Polynominal fitting

    We fit the data using a polynomial function of the form

    .. math::
       y(x, {\mathbf w})=w_{0}+w_{1} x+w_{2} x^{2}+, \cdots,+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}

    Parameters
    ----------
    w : Tensor, optional
        initial coefficient, by default None (generate randomly)
    deg : int, optional
        degree of the Polynominal, by default 1
    trainable : bool, optional
        is ``self.w`` trainable, by default True

    Examples
    --------

    ::

        th.manual_seed(2020)
        Ns, k, b = 100, 1.2, 3.0
        x = th.linspace(0, 1, Ns)
        t = x * k + b + th.randn(Ns)

        deg = (0, 1)

        polyfit = PolyFit(deg=deg)

        lossfunc = th.nn.MSELoss('mean')
        optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, polyfit.parameters()), lr=1e-1)

        for n in range(100):
            y = polyfit(x)

            loss = lossfunc(y, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("---Loss %.4f, %.4f, %.4f" % (loss.item(), polyfit.w[0], polyfit.w[1]))

        # output
        ---Loss 16.7143, -0.2315, -0.1427
        ---Loss 15.5265, -0.1316, -0.0429
        ---Loss 14.3867, -0.0319, 0.0568
        ---Loss 13.2957, 0.0675, 0.1561
        ---Loss 12.2543, 0.1664, 0.2551
                        ...
        ---Loss 0.9669, 2.4470, 1.9995
        ---Loss 0.9664, 2.4515, 1.9967
        ---Loss 0.9659, 2.4560, 1.9938
    """

    def __init__(self, w=None, deg=1, trainable=True):

        super(PolyFit, self).__init__()

        if type(deg) is int:
            deg = (0, deg)
        self.deg = deg
        if w is None:
            self.w = Parameter(th.randn(deg[1] - deg[0] + 1, 1), requires_grad=trainable)
        else:
            self.w = Parameter(w, requires_grad=trainable)

    def forward(self, x):
        y = 0.
        for n in range(self.deg[0], self.deg[1] + 1):
            y = y + self.w[n - self.deg[0]] * (x**n)
        return y


if __name__ == '__main__':

    th.manual_seed(2020)
    Ns, k, b = 100, 1.2, 3.0
    x = th.linspace(0, 1, Ns)
    t = x * k + b + th.randn(Ns)

    deg = (0, 1)

    polyfit = PolyFit(deg=deg)

    lossfunc = th.nn.MSELoss('mean')
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, polyfit.parameters()), lr=1e-1)

    for n in range(100):
        y = polyfit(x)

        loss = lossfunc(y, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("---Loss %.4f, %.4f, %.4f" % (loss.item(), polyfit.w[0], polyfit.w[1]))
