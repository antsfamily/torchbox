#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : cplxfunc.py
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
from torch.nn.functional import relu


def csign(x, cdim=None):
    r"""The signum function like Matlab's sign

    .. math::
        {\rm csign}(x+jy) = \frac{x+jy}{\sqrt{x^2+y^2}}

    Parameters
    ----------
    x : tensor, int, float or complex
        The input
    cdim : int or None, optional
        Specifies the complex axis..

    Returns
    -------
    tensor
        The output.

    Raises
    ------
    TypeError
        :attr:`cdim` should be integer!
    """
    xtype = type(x)
    if (xtype is int) or (xtype is float) or (xtype is complex):
        return x / (abs(x) + EPS)
    if type(x) is not th.Tensor:
        x = th.tensor(x)
    if cdim is None:
        return x / (x.abs() + EPS)
        # return th.sgn(x)
    if type(cdim) is not int:
        raise TypeError('axis should be integer!')
    x = x / (x.pow(2).sum(cdim, keepdim=True).sqrt() + EPS)
    # x = x.transpose(cdim, -1)
    # x = th.view_as_complex(x)
    # x = th.sgn(x)
    # x = th.view_as_real(x)
    # x = x.transpose(cdim, -1)
    return x


def csoftshrink(x, alpha=0.5, cdim=None, inplace=False):
    r"""Complex soft shrink function

    Parameters
    ----------
    x : tensor
        The input.
    alpha : float, optional
        The threshhold.
    cdim : int or None, optional
        Specifies the complex axis.

    Returns
    -------
    tensor
        The output.

    Raises
    ------
    TypeError
        :attr:`cdim` should be integer!
    """
    if cdim is None:
        return csign(x, cdim=cdim) * relu(x.abs() - alpha, inplace=inplace)
    if type(cdim) is not int:
        raise TypeError('axis should be integer!')
    return csign(x, cdim=cdim) * relu(x.pow(2).sum(cdim, keepdim=True).sqrt() - alpha, inplace=inplace)

    # if cdim is None:
    #     x = th.sgn(x) * relu(x.abs() - alpha)
    #     return x
    # if type(cdim) is not int:
    #     raise TypeError('axis should be integer!')
    # x = x.transpose(cdim, -1)
    # x = th.view_as_complex(x)
    # x = th.sgn(x) * relu(x.abs() - alpha)
    # x = th.view_as_real(x)
    # x = x.transpose(cdim, -1))
    # return x


def softshrink(x, alpha=0.5, inplace=False):
    r"""Real soft shrink function

    Parameters
    ----------
    x : tensor
        The input.
    alpha : float, optional
        The threshhold.

    Returns
    -------
    tensor
        The output.
    """
    return th.sgn(x) * relu(x.abs() - alpha, inplace=inplace)


if __name__ == '__main__':

    x = 2 + 2j
    print(csign(x))

    x = [1 + 1j, 2 + 2j]
    print(csign(x))

    print(csoftshrink(th.tensor(x)))
    print(softshrink(th.tensor(x)))
