#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : polynomialfit.py
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
from torch.nn.parameter import Parameter


class PolyFit(th.nn.Module):
    r"""Polynominal fitting

    We fit the data using a polynomial function of the form

    .. math::
       y(x, {\mathbf w})=w_{0}+w_{1} x+w_{2} x^{2}+, \cdots,+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}

    Parameters
    ----------
    w : tensor, optional
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
