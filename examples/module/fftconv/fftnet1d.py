#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th
import torchbox as tb
from torch.nn.parameter import Parameter


class FFTNet1d(th.nn.Module):

    def __init__(self):
        super(FFTNet1d, self).__init__()

        self.f = Parameter(th.tensor(1., requires_grad=True))
        self.fftlayer1d = tb.FFTLayer1d()

    def forward(self, T):

        # X = th.cos(2 * np.pi * self.f * T) + 1j * th.sin(2 * np.pi * self.f * T)

        # Y = self.fftlayer1d(th.view_as_real(X))
        # # Y = th.view_as_complex(Y)

        Xr = th.cos(2 * np.pi * self.f * T)
        Xi = th.sin(2 * np.pi * self.f * T)
        X = Xr + 1j * Xi
        Y = th.fft.fft(X, n=None, dim=0, norm=None)

        return Y
