#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : normalsignals.py
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
from torchbox.utils.const import PI


def rect(x):
    r"""
    Rectangle function:

    .. math::
        rect(x) = {1, if |x|<= 0.5; 0, otherwise}
    """
    # return hs(x + 0.5) * ihs(x - 0.5)
    # return th.where(th.abs(x) > 0.5, 0., 1.0)
    y = th.ones_like(x)
    y[x < -0.5] = 0.
    y[x > 0.5] = 0.
    return y


def chirp(t, T, Kr):
    r"""
    Create a chirp signal:

    .. math::
        S_{tx}(t) = rect(t/T) * exp(1j*pi*Kr*t^2)
    """
    return rect(t / T) * th.exp(1j * PI * Kr * t**2)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    Ts = 3
    Fs = 100
    Ns = int(Ts * Fs)
    x = th.linspace(-Ts / 2., Ts / 2., Ns)

    y = rect(x)

    plt.figure()
    plt.plot(y)
    plt.show()
