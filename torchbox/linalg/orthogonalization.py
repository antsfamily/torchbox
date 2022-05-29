#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 22:29:14
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

from torchbox.utils.const import EPS
import torch as th


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
