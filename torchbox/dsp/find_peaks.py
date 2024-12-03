#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : ffts.py
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

import numpy as np
import torch as th
import torchbox as tb


def localmax1d(data, win=3, thresh=None):
    r"""find local maximum points

    `Pytorch Argrelmax (or C++) function <https://discuss.pytorch.org/t/pytorch-argrelmax-or-c-function/36404/2>`_

    Parameters
    ----------
    data : list, ndarray or Tensor
        the input data
    win : int, optional
        the local window size, by default 3
    thresh : list, ndarray, Tensor or None, optional
        the threshhold, by default :obj:`None`

    Examples
    --------

    ::

        x = th.zeros(100, )
        x[10] = 1.
        x[30] = 1.2
        x[31] = 0.9
        x[80] = 1.
        x[90] = 0.3

        print(localmax1d(x, win=2, thresh=None))
        print(localmax1d(x, win=5))
        print(localmax1d(x, win=5, thresh=0.8))

    """

    if type(data) is not th.Tensor:
        data = th.tensor(data)
    x = data.clone()
    if thresh is not None:
        x[x<thresh] = float('-inf')

    x = x.view(1, 1, -1)
    window_maxima = th.nn.functional.max_pool1d_with_indices(x.view(1, 1, -1), win, 1, padding=win//2)[1].squeeze()
    candidates = window_maxima.unique()
    nice_peaks = candidates[(window_maxima[candidates]==candidates).nonzero()]
    if nice_peaks[0, 0] == 0:
        peak0 = True
        for i in range(1, win):
            if data[i] >= data[0]:
                peak0 = False
                break
        if not peak0:
            nice_peaks = nice_peaks[1:]

    return nice_peaks



if __name__ == "__main__":

    x = th.zeros(100, )
    x[10] = 1.
    x[30] = 1.2
    x[31] = 0.9
    x[80] = 1.
    x[90] = 0.3

    print(localmax1d(x, win=2, thresh=None))
    print(localmax1d(x, win=5))
    print(localmax1d(x, win=5, thresh=0.8))

    x = th.zeros((100, 100))

    x[50, 30] = 1.
    x[20, 80] = 1.
