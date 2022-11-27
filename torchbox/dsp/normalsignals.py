#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : normalsignals.py
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
