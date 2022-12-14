#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : orthogonalization.py
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


def orth(x):
    r"""Orthogonalization

    A function like MATLAB's ``orth``. After orthogonalizing,
    each column is a orthogonal basis.

    Parameters
    ----------
    x : Tensor
        The matrix to be orthogonalized.

    Examples
    --------

    code:
    ::

        x = th.tensor([[1, 2.], [3, 4], [5, 6]])
        y = orth(x)
        print(x)
        print(y)
        print((y[0, :] * y[1, :] * y[2, :]).sum())
        print((y[:, 0] * y[:, 1]).sum())

    result:
    ::

        tensor([[1., 2.],
                [3., 4.],
                [5., 6.]])
        tensor([[-0.2298,  0.8835],
                [-0.5247,  0.2408],
                [-0.8196, -0.4019]])
        tensor(-0.1844)
        tensor(-1.7881e-07)

    """

    u, s, vh = th.linalg.svd(x, full_matrices=False)

    if s is not None:
        # s = th.diag(s)
        tol = max(x.shape) * s[0] * EPS
        r = (s > tol).sum().item()
        u = u[:, 0:r]
    return u


if __name__ == '__main__':

    x = th.tensor([[1, 2.], [3, 4], [5, 6]])
    y = orth(x)
    print(x)
    print(y)
    print((y[0, :] * y[1, :] * y[2, :]).sum())
    print((y[:, 0] * y[:, 1]).sum())
