#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : voptimizer.py
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

    def __init__(self, rankr=0.6, maxrankr=1, mu=1.003):
        r"""

        Initialize Binary optimizer

        Parameters
        ----------
        rankr : float, optional
            the initial proportion :math:`r` of the selected samples (with weights vi=1); (the default is 0.6)
        maxrankr : int, optional
            the upper bound of the annealed sample proportion :math:`r_{max}`. (the default is 1)
        mu : int, optional
            the annealing parameter :math`\mu`, the incremental ratio of the proportion of the selected
            samples in each iteration. (the default is 1.003)
        """
        super(Binary, self).__init__()
        self.spfunction = 'Binary'
        self.lmbd = -1
        self.mu = mu
        self.rankr = rankr
        self.maxrankr = maxrankr
        self.v = -1

    def step(self, loss):
        r"""one step of optimization

        The optimal solution is

        .. math::
           v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
           :label: equ-SPL_BinaryUpdate

        Parameters
        ----------
        loss : tensor
            The loss values of N samples. (:math:`N×1` tensor)
        """

        assert isinstance(loss, th.Tensor)
        N = loss.size(0)
        sortedloss, indices = th.sort(loss, 0, descending=False)
        rankthresh = round(N * self.rankr)
        self.lmbd = (sortedloss[max(rankthresh - 1, 0)] + sortedloss[min(rankthresh, N - 1)]).item() / 2.
        self.lmbd = min(1., self.lmbd)

        self.v = th.ones(N)
        self.v = self.v.to(loss.device)

        self.v[loss > self.lmbd] = 0.
        # self.v[0:4] = 0.
        # print(self.v, "===")

    def update_rankr(self):
        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

        self.rankr = min(self.rankr * self.mu, self.maxrankr)


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

    def __init__(self, rankr=0.6, maxrankr=1, mu=1.003):
        r"""

        Initialize Linear optimizer

        Parameters
        ----------
        rankr : float, optional
            the initial proportion :math:`r` of the selected samples (with weights vi=1); (the default is 0.6)
        maxrankr : int, optional
            the upper bound of the annealed sample proportion :math:`r_{max}`. (the default is 1)
        mu : int, optional
            the annealing parameter :math`\mu`, the incremental ratio of the proportion of the selected
            samples in each iteration. (the default is 1.003)
        """
        super(Linear, self).__init__()
        self.spfunction = 'Linear'
        self.lmbd = -1
        self.mu = mu
        self.rankr = rankr
        self.maxrankr = maxrankr
        self.v = -1

    def step(self, loss):
        r"""one step of optimization

        The optimal solution is

        .. math::
           v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
           :label: equ-SPL_BinaryUpdate

        Parameters
        ----------
        loss : tensor
            The loss values of N samples. (:math:`N×1` tensor)
        """

        assert isinstance(loss, th.Tensor)
        N = loss.size(0)
        sortedloss, indices = th.sort(loss, 0, descending=False)
        rankthresh = round(N * self.rankr)
        self.lmbd = (sortedloss[max(rankthresh - 1, 0)] + sortedloss[min(rankthresh, N - 1)]).item() / 2.
        self.lmbd = min(1., self.lmbd)

        self.v = th.max(1. - loss / self.lmbd, th.tensor(0.))
        self.v = th.min(self.v, th.tensor(1.))

    def update_rankr(self):
        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

        self.rankr = min(self.rankr * self.mu, self.maxrankr)


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

    def __init__(self, rankr=0.6, maxrankr=1, mu=1.003):
        r"""

        Initialize Logarithmic optimizer

        Parameters
        ----------
        rankr : float, optional
            the initial proportion :math:`r` of the selected samples (with weights vi=1); (the default is 0.6)
        maxrankr : int, optional
            the upper bound of the annealed sample proportion :math:`r_{max}`. (the default is 1)
        mu : int, optional
            the annealing parameter :math`\mu`, the incremental ratio of the proportion of the selected
            samples in each iteration. (the default is 1.003)
        """
        super(Logarithmic, self).__init__()
        self.spfunction = 'Logarithmic'
        self.lmbd = -1
        self.mu = mu
        self.rankr = rankr
        self.maxrankr = maxrankr
        self.v = -1

    def step(self, loss):
        r"""one step of optimization

        The optimal solution is

        .. math::
           v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
           :label: equ-SPL_BinaryUpdate

        Parameters
        ----------
        loss : tensor
            The loss values of N samples. (:math:`N×1` tensor)
        """

        assert isinstance(loss, th.Tensor)
        N = loss.size(0)
        sortedloss, indices = th.sort(loss, 0, descending=False)
        rankthresh = round(N * self.rankr)
        self.lmbd = (sortedloss[max(rankthresh - 1, 0)] + sortedloss[min(rankthresh, N - 1)]).item() / 2.
        self.lmbd = min(1., self.lmbd)

        self.v = th.zeros(N)
        self.v = self.v.to(loss.device)
        logsoftidx = (loss < self.lmbd)
        # logsoftidx = logsoftidx * (loss > 0.)

        if abs(1. - self.lmbd) < EPS:
            zeta = EPS
        else:
            zeta = 1. - self.lmbd
        self.v[logsoftidx] = th.log(abs(loss[logsoftidx] + zeta)) / th.log(th.tensor(zeta))
        self.v = th.max(self.v, th.tensor(0.))
        self.v = th.min(self.v, th.tensor(1.))

    def update_rankr(self):
        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

        self.rankr = min(self.rankr * self.mu, self.maxrankr)


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

    def __init__(self, rankr=0.6, maxrankr=1, mu=1.003):
        r"""

        Initialize Mixture optimizer

        Parameters
        ----------
        rankr : float, optional
            the initial proportion :math:`r` of the selected samples (with weights vi=1); (the default is 0.6)
        maxrankr : int, optional
            the upper bound of the annealed sample proportion :math:`r_{max}`. (the default is 1)
        mu : int, optional
            the annealing parameter :math`\mu`, the incremental ratio of the proportion of the selected
            samples in each iteration. (the default is 1.003)
        """
        super(Mixture, self).__init__()
        self.spfunction = 'Mixture'
        self.lmbd = -1
        self.lmbd = -1
        self.mu = mu
        self.rankr = rankr
        self.maxrankr = maxrankr
        self.v = -1

    def step(self, loss):
        r"""one step of optimization

        The optimal solution is

        .. math::
           v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
           :label: equ-SPL_BinaryUpdate

        Parameters
        ----------
        loss : tensor
            The loss values of N samples. (:math:`N×1` tensor)
        """

        assert isinstance(loss, th.Tensor)
        N = loss.size(0)
        sortedloss, indices = th.sort(loss, 0, descending=False)
        rankthresh = round(N * self.rankr)
        self.lmbd = (sortedloss[max(rankthresh - 1, 0)] + sortedloss[min(rankthresh, N - 1)]).item() / 2.
        # self.lmbd = min(1., self.lmbd)

        # print(self.lmbd, "]]]")
        self.lmbd2 = self.lmbd / 2.
        zeta = self.lmbd2 * self.lmbd / (self.lmbd - self.lmbd2)

        self.v = th.ones(N)
        self.v = self.v.to(loss.device)

        self.v[loss > self.lmbd] = 0.
        self.v[loss < self.lmbd2] = 1.
        mixidx = (loss >= self.lmbd2)
        mixidx = mixidx * (loss <= self.lmbd)
        self.v[mixidx] = zeta / loss[mixidx] - zeta / self.lmbd

    def update_rankr(self):
        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

        self.rankr = min(self.rankr * self.mu, self.maxrankr)


if __name__ == '__main__':

    import torchbox as tb

    loss = th.randn(10)

    print("=========Binary==========")

    voptimizer = tb.voptimizer.Binary()
    print("---v,", voptimizer.v)

    print("---loss,", loss)
    voptimizer.step(loss)

    print("---v,", voptimizer.v)

    print("=========Linear==========")

    voptimizer = tb.voptimizer.Linear()
    print("---v,", voptimizer.v)

    print("---loss,", loss)
    voptimizer.step(loss)

    print("---v,", voptimizer.v)

    print("=========Logarithmic==========")

    voptimizer = tb.voptimizer.Logarithmic()
    print("---v,", voptimizer.v)

    print("---loss,", loss)
    voptimizer.step(loss)

    print("---v,", voptimizer.v)

    print("========Mixture===========")

    voptimizer = tb.voptimizer.Mixture()
    print("---v,", voptimizer.v)

    print("---loss,", loss)
    voptimizer.step(loss)

    print("---v,", voptimizer.v)
