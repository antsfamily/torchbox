#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : orthogonalization.py
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
