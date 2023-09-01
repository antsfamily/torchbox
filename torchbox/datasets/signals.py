#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : mnist.py
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
import torchbox as tb


def cosine(t, alpha=1., freq=1., phi=0., reduction=None):
    """generates cosine signal

    Parameters
    ----------
    t : Tensor
        the discretized time steps
    alpha : float, tensor or list, optional
        amplitudes, by default 1.
    freq : float, tensor or list, optional
        frequencies, by default 1.
    phi : float, tensor or list, optional
        initial phases, by default 0.
    reduction : str or None, optional
        ``'mean'``, ``'sum'`` or None
    """

    if type(alpha) is not th.Tensor:
        alpha = th.tensor(alpha)
    if type(freq) is not th.Tensor:
        freq = th.tensor(freq)
    if type(phi) is not th.Tensor:
        phi = th.tensor(phi)

    N = len(phi)
    if len(t) != N:
        t = t.repeat(len(phi), 1)
    alpha, freq, phi = alpha.reshape(N, 1), freq.reshape(N, 1), phi.reshape(N, 1)

    x = alpha * th.cos(2 * th.pi * freq * t + phi)
    if reduction in ['sum', 'SUM']:
        x = th.sum(x, dim=0)
    elif reduction in ['mean', 'MEAN']:
        x = th.mean(x, dim=0)
    elif reduction is not None:
        raise ValueError('Not support reduction mode: %s' % reduction)
    
    return x

class CosineTask:
    def __init__(self, ntask=10, arange=[0.1, 50.], frange=[3., 10.], prange=[0, 3.1415926], trange=[-0.5, 0.5, 500], tmode='uniform', seed=None) -> None:
        self.ntask = ntask
        self.arange = arange
        self.frange = frange
        self.prange = prange
        self.trange = trange
        self.tmode = tmode

        tb.setseed(seed, target='torch')
        self.alpha = tb.scale(th.rand((ntask,)), st=self.arange, sf=[0., 1.])
        self.freq = tb.scale(th.rand((ntask,)), st=self.frange, sf=[0., 1.])
        self.phi = tb.scale(th.rand((ntask,)), st=self.prange, sf=[0., 1.])
        if self.tmode in ['uniform', 'UNIFORM']:
            self.t = tb.scale(th.rand((ntask, trange[2])), st=trange[:-1], sf=[0., 1.])
        if self.tmode in ['sequential', 'SEQUENTIAL']:
            self.t = th.linspace(trange[0], trange[1], trange[2]).repeat(ntask, 1)

    def mktask(self, ntask=10, arange=None, frange=None, prange=None, trange=None, seed=None, device=None, rett=False):

        tb.setseed(seed, target='torch')
        alpha = self.alpha if arange is None else tb.scale(th.rand((ntask,)), st=self.arange, sf=[0., 1.])
        freq = self.freq if frange is None else tb.scale(th.rand((ntask,)), st=self.frange, sf=[0., 1.])
        phi = self.phi if prange is None else tb.scale(th.rand((ntask,)), st=self.prange, sf=[0., 1.])

        if trange is None:
            t = self.t
        else:
            if self.tmode in ['uniform', 'UNIFORM']:
                t = tb.scale(th.rand((ntask, trange[2])), st=trange[:-1], sf=[0., 1.])
            if self.tmode in ['sequential', 'SEQUENTIAL']:
                t = th.linspace(trange[0], trange[1], trange[2]).repeat(ntask, 1)

        if rett:
            return cosine(t, alpha=alpha, freq=freq, phi=phi, reduction=None), t
        else:
            return cosine(t, alpha=alpha, freq=freq, phi=phi, reduction=None)


if __name__ == '__main__':

    import torchbox as tb

    freq = [10, 30.]
    alpha = [10, 6.]
    phi = [0, 0.]
    Ts = 1.
    Fs1, Fs2 = 80, 500
    Ns1, Ns2 = int(Fs1*Ts), int(Fs2*Ts)
    t1 = th.linspace(-Ts/2., Ts/2., Ns1)
    t2 = th.linspace(-Ts/2., Ts/2., Ns2)
    x1 = tb.cosine(t1, alpha=alpha, freq=freq, phi=phi)
    x2 = tb.cosine(t2, alpha=alpha, freq=freq, phi=phi)
    print(x1.shape, x2.shape)

    plt = tb.plot([[x1[0], x1[1]], [x2[0], x2[1]]], Xs=[[t1, t1], [t2, t2]], grids=True, xlabels=['Time/s', 'Time/s'])

    ntask = 10
    costask = CosineTask(arange=[0.1, 50.], frange=[10., 10.], prange=[0, 0.], trange=[-0.5, 0.5, 500], tmode='sequential')
    x, t = costask.mktask(ntask=ntask, device='cuda:0', rett=True)
    print(x.shape, t.shape)
    plt = tb.plot([[x[i].cpu() for i in range(ntask)]], Xs=[[t[i].cpu() for i in range(ntask)]], grids=True, xlabels=['Time/s'])
    plt.show()

