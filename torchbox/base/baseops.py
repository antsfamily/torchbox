#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : baseops.py
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
import copy


def upkeys(D, mode='-', k='module.'):
    r"""update keys of a dictionary

    Parameters
    ----------
    D : dict
        the input dictionary
    mode : str, optional
        ``'-'`` for remove key string which is specified by :attr:`k`, by default '-'
        ``'+'`` for add key string which is specified by :attr:`k`, by default '-'
    k : str, optional
        key string pattern, by default 'module.'

    Returns
    -------
    dict
        new dictionary with keys updated
    """
    
    X = {}
    for key, value in D.items():
        if mode == '-':
            newkey = key.replace(k, '')
        if mode == '+':
            newkey = k + key
        X[newkey] = value
    
    return X


def dreplace(d, fv=None, rv='None', new=False):
    """replace dict value

    Parameters
    ----------
    d : dict
        the dict
    fv : any, optional
        to be replaced, by default None
    rv : any, optional
        replaced with, by default 'None'
    new : bool, optional
        if true, deep copy dict, will not change input, by default False

    Returns
    -------
    dict
        dict with replaced value
    """
    
    fvtype = type(fv)
    if new:
        d = copy.deepcopy(d)
    for k, v in d.items():
        if type(v) is dict:
            dreplace(v, fv=fv, rv=rv)
        else:
            if type(v) == fvtype:
                if v == fv:
                    d[k] = rv
    return d


def dmka(D, Ds):
    r"""Multiple key-value assign to a dict

    Parameters
    ----------
    D : dict
        main dict
    Ds : dict
        sub dict

    Returns
    -------
    dict
        after assign
    """

    for k, v in Ds.items():
        D[k] = v
    return D


def cat(shapes, axis=0):
    r"""Concatenates

    Concatenates the given sequence of seq shapes in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    Parameters
    ----------
    shapes : tuples or lists
        (shape1, shape2, ...)
    axis : int, optional
        specify the concatenated axis (the default is 0)

    Returns
    -------
    tuple or list
        concatenated shape

    Raises
    ------
    ValueError
        Shapes are not consistent in axises except the specified one.
    """

    x = 0
    s = copy.copy(shapes[0])
    s = list(s)
    for shape in shapes:
        for ax in range(len(s)):
            if (ax != axis) and (s[ax] != shape[ax]):
                raise ValueError("All tensors must either have \
                    the same shape (except in the concatenating dimension)\
                     or be empty.")
        x += shape[axis]
        # print(x)
    # print(s, x)
    s[axis] = x
    return s


if __name__ == '__main__':
    import torchbox as tb
    import torch as th

    D = {'a': 1, 'b': 2, 'c': 3}
    Ds = {'b': 6}
    print(D)
    dmka(D, Ds)
    print(D)

    x = th.randn(2, 3)
    xs = x.shape
    xs = list(xs)
    print(xs)
    print('===cat')
    print(x.size())
    print('---Theoretical result')

    ys = tb.cat((xs, xs, xs), 0)
    print(ys)

    ys = tb.cat((xs, xs, xs), 1)
    print(ys)
    print('---Torch result')

    y = th.cat((x, x, x), 0)
    print(y.size())
    y = th.cat((x, x, x), 1)
    print(y.size())
