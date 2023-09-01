#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : baseops.py
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

import copy
import numpy as np
import torch as th


def dimpos(ndim, dim):
    """make positive dimensions

    Parameters
    ----------
    ndim : int
        the number of dimensions
    dim : int, list or tuple
        the dimension index to be converted
    """

    if type(dim) is int:
        newdim = dim + ndim if dim < 0 else dim
    elif type(dim) in [list, tuple]:
        newdim = [d + ndim if d < 0 else d for d in dim]
    return newdim


def rmcdim(ndim, cdim, dim, keepdim):
    r"""get dimension indexes after removing cdim

    Parameters
    ----------
    ndim : int
        the number of dimensions
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to :obj:`None`
    dim : int, None, tuple or list
        dimensions to be re-defined
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)

    Returns
    -------
    int, tuple or list
         re-defined dimensions
        
    """

    if cdim is None:
        return dim

    if cdim < 0:
        cdim = ndim + cdim

    if dim is None:
        newdim = list(range(ndim))
        newdim.remove(cdim)
        return newdim
    else:
        if keepdim:
            return dim
        else:
            if type(dim) is int:
                dim = dim if dim >= 0 else ndim + dim
                newdim = dim if cdim > dim else dim - 1
            else:
                newdim = []
                for d in dim:
                    d = d if d >= 0 else ndim + d
                    newdim.append(d if cdim > d else d - 1)
            return newdim

def dimpermute(ndim, dim, mode=None, dir='f'):
    """permutes dimensions

    Parameters
    ----------
    ndim : int
        the number of dimensions
    dim : list or tuple
        the order of new dimensions (:attr:`mode` is :obj:`None`) or multiplication dimensions (``'matmul'``)
    mode : str or None, optional
        permution mode, ``'matmul'`` for matrix multiplication, 
        ``'merge'`` for dimension merging (putting the dimensions specified by second and subsequent elements of :attr:`dim`
        after the dimension specified by the specified by the first element of :attr:`dim`), 
        :obj:`None` for regular permute, such as torch.permute, by default :obj:`None`.
    dir : str, optional
        the direction, ``'f'`` or ``'b'`` (reverse process of ``'f'``), default is ``'f'``.
    """
    
    dim = dimpos(ndim, dim)

    if (mode is None) or (mode.lower() == ''):
        newdim = dim
    elif mode.lower() == 'merge':
        newdim = list(range(ndim))
        if type(dim) is int:
            return newdim
        for d in dim[1:]:
            newdim.remove(d)
        pos0 = newdim.index(dim[0])
        for k, d in enumerate(dim[1:]):
            newdim.insert(pos0+k+1, d)
    elif mode.lower() == 'matmul':
        if len(dim) != 2:
            raise ValueError('For matrix multiplication, dim should has length 2')
        newdim = list(range(ndim))
        for d in dim:
            newdim.remove(d)
        newdim += dim
    else:
        raise ValueError('Not supported mode: %s' % mode)

    if dir.lower() == 'f':
        return newdim
    elif dir.lower() == 'b':
        newdim = sorted(range(len(newdim)), key=lambda k: newdim[k], reverse=False)
        return newdim
    else:
        raise ValueError('Not supported dir: %s' % dir)

def dimreduce(ndim, cdim, dim, keepcdim=False, reduction=None):
    """get dimensions for reduction operation

    Parameters
    ----------
    ndim : int
        the number of dimensions
    cdim : int, optional
        if the data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to :obj:`None`
    dim : int, list, tuple or None
        dimensions for processing, :obj:`None` means all
    keepcdim : bool
        keep the complex dimension? The default is :obj:`False`
    reduction : str or None, optional
        The operation in other dimensions except the dimensions specified by :attr:`dim`,
        ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    """    

    dims = list(range(ndim))
    if cdim is not None:
        if cdim < 0:
            cdim = cdim + ndim
    if dim is None:
        dim = dims.copy()
    elif type(dim) is int:
        dim = [dim]
    else:
        dim = list(dim)
    
    if reduction is None:
        if cdim is not None:
            dims = dim if keepcdim else dim + [cdim]
        else:
            dims = dim
    else:
        if (cdim is not None) and (keepcdim is True):
            dims.remove(cdim)
    
    return dims


def dimmerge(ndim, mdim, dim, keepdim=False):
    """obtain new dimension indexes after merging

    Parameters
    ----------
    ndim : int
        the number of dimensions
    mdim : int, list or tuple
        the dimension indexes for merging
    dim : int, list or tuple
        the dimension indexes that are not merged
    keepdim : bool
        keep the dimensions when merging?
    """

    if (type(mdim) is int) or keepdim:
        return mdim, dim
    else:
        mdim = dimpos(ndim, mdim)
        dim = dimpos(ndim, dim)
        flag = [' '] * ndim
        if type(dim) is int:
            flag[dim] = 'dim'
        else:
            for d in dim:
                flag[d] = 'dim%d' % d
        for k, d in enumerate(mdim):
            flag[d] = 'mdim%d' % k
        newflag = []
        for f in flag:
            if ('mdim' not in f) or (f == 'mdim0'):
                newflag.append(f)
        
        if type(dim) is int:
            newdim = newflag.index('dim')
        else:
            newdim = []
            for d in dim:
                newdim.append(newflag.index('dim%d' % d))

        return newflag.index('mdim0'), newdim


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
    s[axis] = x
    return s


def argsort(x, reverse=False):
    r"""returns index of sorted array

    Parameters
    ----------
    x : list, ndarray or tensor
        the input
    reverse : bool, optional
        sort in reversed order?, by default False

    Returns
    -------
    list, ndarray or tensor
        the index
    """

    idx = sorted(range(len(x)), key=lambda k: x[k], reverse=reverse)

    if type(x) is th.Tensor:
        idx = th.tensor(idx)
    elif type(x) is np.ndarray:
        idx = np.array(idx)
    else:
        pass

    return idx
    

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

    print(rmcdim(4, dim=(1, 2), cdim=3, keepdim=True))
    print(rmcdim(4, dim=(1, 2), cdim=3, keepdim=False))
    print(rmcdim(4, dim=(1, 2), cdim=-1, keepdim=False))
    print(rmcdim(4, dim=(1, -2), cdim=-1, keepdim=False))
    print(rmcdim(4, dim=(0, 2, 3), cdim=1, keepdim=True))
    print(rmcdim(4, dim=(0, 2, 3), cdim=1, keepdim=False))
    print(rmcdim(4, dim=(0, 2, 3), cdim=-3, keepdim=False))
    print(rmcdim(4, dim=(0, -2, 3), cdim=-3, keepdim=False))  # 0, 1, 2, 3
    print(rmcdim(4, dim=-2, cdim=-3, keepdim=False))  # 0, 1, 2, 3
    print(rmcdim(4, dim=0, cdim=-3, keepdim=False))  # 0, 1, 2, 3
    print(rmcdim(4, dim=3, cdim=-3, keepdim=False))  # 0, 1, 2, 3
    print(rmcdim(4, dim=-1, cdim=-3, keepdim=False))  # 0, 1, 2, 3

    print(argsort([3, 2, 1, 0], reverse=False))
