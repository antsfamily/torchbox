#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : spfunction.py
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
import torch as th
from torchbox.utils.const import EPS


class Binary(object):
    r"""binary function

    The binary SPL function can be expressed as

    .. math::
       f(\bm{v}, k) =  = -lambd\|{\bm v}\|_1 = -lambd\sum_{n=1}^N v_n
       :label: equ-SPL_BinaryFunction

    The optimal solution is

    .. math::
       v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
       :label: equ-SPL_BinaryUpdate

    """

    def __init__(self):
        r"""

        Initialize Binary SPfunction

        """
        super(Binary, self).__init__()
        self.name = 'Binary'

    def eval(self, v, lmbd):
        r"""eval SP function

        The binary SPL function can be expressed as

        .. math::
           f(\bm{v}, k) =  = -lambd\|{\bm v}\|_1 = -lambd\sum_{n=1}^N v_n
           :label: equ-SPL_BinaryFunction

        Parameters
        ----------
        v : tensor
            The easy degree of N samples. (:math:`N×1` tensor)
        lmbd : float
            balance factor
        """

        assert isinstance(v, th.Tensor)

        return -lmbd * th.sum(v)


class Linear(object):
    r"""Linear function

    The Linear SPL function can be expressed as

    .. math::
       f(\bm{v}, \lambda)=\lambda\left(\frac{1}{2}\|\bm{v}\|_{2}^{2}-\sum_{n=1}^{N} v_{n}\right)
       :label: equ-SPL_LinearFunction

    The optimal solution is

    .. math::
       v_{n}^* = {\rm max}\{1-l_n/\lambda, 0\}
       :label: equ-SPL_LinearUpdate

    """

    def __init__(self):
        r"""

        Initialize Linear SPfunction

        """
        super(Linear, self).__init__()
        self.name = 'Linear'

    def eval(self, v, lmbd):
        r"""eval SP function

        The Linear SPL function can be expressed as

        .. math::
           f(\bm{v}, \lambda)=\lambda\left(\frac{1}{2}\|\bm{v}\|_{2}^{2}-\sum_{n=1}^{N} v_{n}\right)
           :label: equ-SPL_LinearFunction

        Parameters
        ----------
        v : tensor
            The easy degree of N samples. (:math:`N×1` tensor)
        lmbd : float
            balance factor
        """

        assert isinstance(v, th.Tensor)

        return lmbd * (0.5 * th.sum(v**2) - th.sum(v))


class Logarithmic(object):
    r"""Logarithmic function

    The Logarithmic SPL function can be expressed as

    .. math::
       f(\bm{v}, \lambda) = \sum_{n=1}^{N}\left(\zeta v_{n}-\frac{\zeta^{v_{n}}}{{\rm log} \zeta}\right)
       :label: equ-SPL_LogarithmicFunction

    where, :math:`\zeta=1-\lambda, 0<\lambda<1`

    The optimal solution is

    .. math::
       v_{n}^{*}=\left\{\begin{array}{ll}{0,} & {l_{n}>=\lambda} \\ {\log \left(l_{n}+\zeta\right) / \log \xi,} & {l_{n}<\lambda}\end{array}\right.
       :label: equ-SPL_LogarithmicUpdate

    """

    def __init__(self):
        r"""

        Initialize Logarithmic SPfunction

        """
        super(Logarithmic, self).__init__()
        self.name = 'Logarithmic'

    def eval(self, v, lmbd):
        r"""eval SP function

        The Logarithmic SPL function can be expressed as

        .. math::
           f(\bm{v}, \lambda) = \sum_{n=1}^{N}\left(\zeta v_{n}-\frac{\zeta^{v_{n}}}{{\rm log} \zeta}\right)
           :label: equ-SPL_LogarithmicFunction

        where, :math:`\zeta=1-\lambda, 0<\lambda<1`

        Parameters
        ----------
        v : tensor
            The easy degree of N samples. (:math:`N×1` tensor)
        lmbd : float
            balance factor
        """

        assert isinstance(v, th.Tensor)

        zeta = 1. - lmbd
        return th.sum(zeta * v - zeta**v / np.log(zeta))


class Mixture(object):
    r"""Mixture function

    The Mixture SPL function can be expressed as

    .. math::
       f\left(\bm{v}, lambd \right)=-\zeta \sum_{n=1}^{N} \log \left(v_{n}+\zeta / lambd \right)
       :label: equ-SPL_MixtureFunction

    where, :math:`ζ= \frac{1}{k^{\prime} - k} = \frac{\lambda^{\prime}\lambda}{\lambda-\lambda^{\prime}}`

    The optimal solution is

    .. math::
       v_{n}^{*}=\left\{\begin{array}{ll}{1,} & {l_{n} \leq \lambda^{\prime}} \\ {0,} & {l_{n} \geq \lambda} \\ {\zeta / l_{n}-\zeta / \lambda,} & {\text { otherwise }}\end{array}\right.
       :label: equ-SPL_MixtureUpdate

    """

    def __init__(self):
        r"""

        Initialize Mixture SPfunction

        """
        super(Mixture, self).__init__()
        self.name = 'Mixture'

    def eval(self, v, lmbd1, lmbd2):
        r"""eval SP function

        The Mixture SPL function can be expressed as

        .. math::
           f\left(\bm{v}, lambd \right)=-\zeta \sum_{n=1}^{N} \log \left(v_{n}+\zeta / lambd \right)
           :label: equ-SPL_MixtureFunction

        where, :math:`ζ= \frac{1}{k^{\prime} - k} = \frac{\lambda^{\prime}\lambda}{\lambda-\lambda^{\prime}}`


        Parameters
        ----------
        v : tensor
            The easy degree of N samples. (:math:`N×1` tensor)
        """

        assert isinstance(v, th.Tensor)

        zeta = lmbd2 * lmbd1 / (lmbd1 - lmbd2)
        return -zeta * th.mean(th.log(v + zeta / lmbd1))


if __name__ == '__main__':

    import torchbox as tb

    loss = th.randn(10)

    print("=========Binary==========")

    SPfunction = tb.spfunction.Binary()
    Voptimizer = tb.voptimizer.Binary()
    Voptimizer.step(loss)
    fv = SPfunction.eval(Voptimizer.v, Voptimizer.lmbd)
    print("---fv,", fv)

    print("=========Linear==========")

    SPfunction = tb.spfunction.Linear()
    Voptimizer = tb.voptimizer.Linear()
    Voptimizer.step(loss)
    fv = SPfunction.eval(Voptimizer.v, Voptimizer.lmbd)
    print("---fv,", fv)

    print("=========Logarithmic==========")

    SPfunction = tb.spfunction.Logarithmic()
    Voptimizer = tb.voptimizer.Logarithmic()
    Voptimizer.step(loss)
    fv = SPfunction.eval(Voptimizer.v, Voptimizer.lmbd)
    print("---fv,", fv)

    print("========Mixture===========")

    SPfunction = tb.spfunction.Mixture()
    Voptimizer = tb.voptimizer.Mixture()
    Voptimizer.step(loss)
    fv = SPfunction.eval(Voptimizer.v, Voptimizer.lmbd, Voptimizer.lmbd2)
    print("---fv,", fv)
