#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : transform.py
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
from torchbox.utils.const import EPS


class Standardization(th.nn.Module):
    r"""Standardization

    .. math::
        \bar{X} = \frac{X-\mu}{\sigma}


    Parameters
    ----------
    X : Tensor
        data to be normalized,
    mean : list or None, optional
        mean value (the default is None, which means auto computed)
    std : list or None, optional
        Standard deviation (the default is None, which means auto computed)
    axis : list or int, optional
        Specify the axis for computing mean and standard deviation (the default is None, which means all elements)
    unbiased : bool, optional
        If unbiased is False, then the standard-deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.
    retall : bool, optional
        If True, also return the mean and std (the default is False, which means just return the standardized data)

    Examples
    ----------

    ::

        import torchbox as tb
        tb.setseed(seed=2020, target='torch')
        x = th.randn(5, 2, 4, 3)

        f = Standardization(axis=(2, 3), unbiased=False, retall=True)
        y, meanv, stdv = f(x)
        print(y[0], y.shape)

        g = th.nn.InstanceNorm2d(2)

        z = g(x)
        print(z[0], z.shape)

        f = Standardization(axis=(0, 2, 3), unbiased=False, retall=True)
        y, meanv, stdv = f(x)
        print(y[0], y.shape)

        g = th.nn.BatchNorm2d(2)

        z = g(x)
        print(z[0], z.shape)

    The results are: ::

        tensor([[[ 0.2761, -0.1161, -1.3316],
                 [ 0.4918,  0.5450, -0.7350],
                 [ 1.5699, -1.8567,  1.7366],
                 [-0.1463, -0.1318, -0.3019]],

                [[-1.0576,  0.5794, -0.6489],
                 [-0.3410, -1.6589,  0.2531],
                 [ 1.2150,  0.7262,  0.3333],
                 [-1.1270, -0.2132,  1.9397]]]) torch.Size([5, 2, 4, 3])
        tensor([[[ 0.2761, -0.1161, -1.3316],
                 [ 0.4918,  0.5450, -0.7350],
                 [ 1.5699, -1.8567,  1.7366],
                 [-0.1463, -0.1318, -0.3019]],

                [[-1.0576,  0.5794, -0.6489],
                 [-0.3410, -1.6588,  0.2531],
                 [ 1.2150,  0.7262,  0.3333],
                 [-1.1270, -0.2132,  1.9397]]]) torch.Size([5, 2, 4, 3])
        tensor([[[ 0.0498, -0.2576, -1.2101],
                 [ 0.2188,  0.2605, -0.7426],
                 [ 1.0637, -1.6216,  1.1943],
                 [-0.2812, -0.2698, -0.4032]],

                [[-1.1965,  0.0760, -0.8788],
                 [-0.6395, -1.6639, -0.1776],
                 [ 0.5701,  0.1901, -0.1153],
                 [-1.2505, -0.5402,  1.1335]]]) torch.Size([5, 2, 4, 3])
        tensor([[[ 0.0498, -0.2576, -1.2101],
                 [ 0.2188,  0.2605, -0.7426],
                 [ 1.0637, -1.6216,  1.1943],
                 [-0.2812, -0.2698, -0.4032]],

                [[-1.1965,  0.0760, -0.8788],
                 [-0.6395, -1.6639, -0.1776],
                 [ 0.5701,  0.1901, -0.1153],
                 [-1.2505, -0.5401,  1.1335]]], grad_fn=<SelectBackward>) torch.Size([5, 2, 4, 3])

    """

    def __init__(self, mean=None, std=None, axis=None, unbiased=False, retall=False):
        super(Standardization, self).__init__()
        self.mean = mean
        self.std = std
        self.axis = axis
        self.unbiased = unbiased
        self.retall = retall

    def forward(self, x):

        if self.mean is None:
            if self.axis is None:
                mean = th.mean(x)
            else:
                mean = th.mean(x, self.axis, keepdim=True)

        if self.std is None:
            if self.axis is None:
                std = th.std(x, self.unbiased)
            else:
                std = th.std(x, self.axis, self.unbiased, keepdim=True)

        if self.retall is True:
            return (x - mean) / (std + EPS), mean, std
        else:
            return (x - mean) / (std + EPS)


if __name__ == '__main__':

    import torchbox as tb
    tb.setseed(seed=2020, target='torch')
    x = th.randn(5, 2, 4, 3) * 10000

    f = Standardization(axis=(2, 3), unbiased=False, retall=True)
    y, meanv, stdv = f(x)
    print(y[0], y.shape, meanv.shape, stdv.shape)

    g = th.nn.InstanceNorm2d(2)

    z = g(x)
    print(z[0], z.shape)

    f = Standardization(axis=(0, 2, 3), unbiased=False, retall=True)
    y, meanv, stdv = f(x)
    print(y[0], y.shape, meanv.shape, stdv.shape)

    g = th.nn.BatchNorm2d(2)

    z = g(x)
    print(z[0], z.shape)
