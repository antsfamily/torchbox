#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : arrayops.py
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


def sl(dims, axis, idx=None, **kwargs):
    r"""Slice any axis

    generates slice in specified axis.

    Parameters
    ----------
    dims : int
        total dimensions
    axis : int or list
        select axis list.
    idx : list or None, optional
        slice lists of the specified :attr:`axis`, if None, does nothing (the default)
    dim : int or list
        (kwargs) if specified, will overwrite :attr:`axis`

    Returns
    -------
    tuple of slice
        slice for specified axis elements.

    Examples
    --------

    ::

        import numpy as np

        np.random.seed(2020)
        X = np.random.randint(0, 100, (9, 10))
        print(X, 'X)
        print(X[sl(2, -1, [0, 1])], 'Xsl')

        # output:

        [[96  8 67 67 91  3 71 56 29 48]
        [32 24 74  9 51 11 55 62 67 69]
        [48 28 20  8 38 84 65  1 79 69]
        [74 73 62 21 29 90  6 38 22 63]
        [21 68  6 98  3 20 55  1 52  9]
        [83 82 65 42 66 55 33 80 82 72]
        [94 91 14 14 75  5 38 83 99 10]
        [80 64 79 30 84 22 46 26 60 13]
        [24 63 25 89  9 69 47 89 55 75]] X
        [[96  8]
        [32 24]
        [48 28]
        [74 73]
        [21 68]
        [83 82]
        [94 91]
        [80 64]
        [24 63]] Xsl
    """

    if 'dim' in kwargs:
        axis = kwargs['dim']

    idxall = [slice(None)] * dims

    axis = [axis] if type(axis) is int else axis
    idx = [idx] if type(idx) not in [list, tuple] else idx
    if len(axis) != len(idx):
        raise ValueError('The index for each axis should be given!')

    naxis = len(axis)
    for n in range(naxis):
        idxall[axis[n]] = idx[n]

    return tuple(idxall)


def cut(x, pos, axis=None, **kwargs):
    r"""Cut array at given position.

    Cut array at given position.

    Parameters
    ----------
    x : array or tensor
        a tensor to be cut
    pos : tuple or list
        cut positions: ((cpstart, cpend), (cpstart, cpend), ...)
    axis : int, tuple or list, optional
        cut axis (the default is None, which means nothing)
    """

    if 'dim' in kwargs:
        axis = kwargs['dim']

    if axis is None:
        return x
    if type(axis) == int:
        axis = tuple([axis])
    nDims = x.dim()
    idx = [None] * nDims

    if len(axis) > 1 and len(pos) != len(axis):
        raise ValueError('You should specify cut axis for each cut axis!')
    elif len(axis) == 1:
        axis = tuple(list(axis) * len(pos))

    uqaixs = np.unique(axis)
    for a in uqaixs:
        idx[a] = []

    for i in range(len(axis)):
        idx[axis[i]] += range(pos[i][0], pos[i][1])

    for a in uqaixs:
        idxall = [slice(None)] * nDims
        idxall[a] = idx[a]
        x = x[tuple(idxall)]
    return x


def arraycomb(arrays, out=None):
    r"""compute the elemnts combination of several lists.

    Args:
        arrays (list or tensor): The lists or tensors.
        out (tensor, optional): The combination results (defaults is :obj:`None`).

    Returns:
        tensor: The combination results.

    Examples:

    Compute the combination of three lists: :math:`[1,2,3]`, :math:`[4, 5]`, :math:`[6,7]`,
    this will produce a :math:`12\times 3` array.

    ::

        x = arraycomb(([1, 2, 3], [4, 5], [6, 7]))
        print(x, x.shape)

        # output:
        [[1 4 6]
        [1 4 7]
        [1 5 6]
        [1 5 7]
        [2 4 6]
        [2 4 7]
        [2 5 6]
        [2 5 7]
        [3 4 6]
        [3 4 7]
        [3 5 6]
        [3 5 7]] (12, 3)

    """
    arrays = [x if type(x) is th.Tensor else th.tensor(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.numel() for x in arrays])
    if out is None:
        out = th.zeros([n, len(arrays)], dtype=dtype)
    m = int(n / arrays[0].numel())
    out[:, 0] = arrays[0].repeat_interleave(m)

    if arrays[1:]:
        arraycomb(arrays[1:], out=out[0:m, 1:])

    for j in range(1, arrays[0].numel()):
        out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]

    return out


def pmutdims(ndims, dims, mode=None, dir='f'):
    """permutes axes

    Parameters
    ----------
    ndims : int
        the number of dimensions
    dims : list or tuple
        the order of new dimensions (:attr:`mode` is :obj:`None`) or multiplication dimensions (``'matmul'``)
    mode : str or None, optional
        permution mode, ``'matmul'`` for matrix multiplication, :obj:`None` for regular permute, such as torch.permute, by default :obj:`None`.
    dir : str, optional
        the direction, ``'f'`` or ``'b'`` (reverse process of ``'f'``), default is ``'f'``.
    """
    
    dims = [d + ndims if d<0 else d for d in dims]

    if (mode is None) or (mode.lower() == ''):
        if dir.lower() == 'f':
            return dims
        elif dir.lower() == 'b':
            newdims = sorted(range(len(dims)), key=lambda k: dims[k], reverse=False)
            return newdims
        else:
            raise ValueError('Not supported dir: %s' % dir)
         
    elif mode.lower() == 'matmul':
        if len(dims) != 2:
            raise ValueError('For matrix multiplication, dims should has length 2')
        newdims = alldims = list(range(ndims))
        if dir.lower() == 'f':
            for d in dims:
                newdims.remove(d)
            newdims += dims
            return newdims
        elif dir.lower() == 'b':
            newdims = alldims[:-2]
            for d1, d2 in zip(dims, alldims[-2:]):
                newdims.insert(d1, d2)
            return newdims
    else:
        raise ValueError('Not supported mode: %s' % mode)


def permute(X, dims, mode=None, dir='f'):
    """permutes axes of tensor

    Parameters
    ----------
    X : tensor
        the input tensor
    dims : list or tuple
        the order of new dimensions (:attr:`mode` is :obj:`None`) or multiplication dimensions (``'matmul'``)
    mode : str or None, optional
        permution mode, ``'matmul'`` for matrix multiplication, :obj:`None` for regular permute, such as torch.permute, by default None.
    dir : str, optional
        the direction, ``'f'`` or ``'b'`` (reverse process of ``'f'``), default is ``'f'``.
    """    
    
    dims = [d + X.ndim if d<0 else d for d in dims]

    return X.permute(pmutdims(X.ndim, dims=dims, mode=mode, dir=dir))


if __name__ == '__main__':

    X = th.randint(0, 100, (9, 10))
    print('X')
    print(X)
    Y = cut(X, ((1, 4), (5, 8)), axis=0)
    print('Y = cut(X, ((1, 4), (5, 8)), axis=0)')
    print(Y)
    Y = cut(X, ((1, 4), (7, 9)), axis=(0, 1))
    print('Y = cut(X, ((1, 4), (7, 9)), axis=(0, 1))')
    print(Y)
    Y = cut(X, ((1, 4), (1, 4), (5, 8), (7, 9)), axis=(0, 1, 0, 1))
    print('cut(X, ((1, 4), (1, 4), (5, 8), (7, 9)), axis=(0, 1, 0, 1))')
    print(Y)

    print(X[sl(2, -1, [[0, 1]])])
    print(X[:, 0:2])

    x = arraycomb(([1, 2, 3, 4], [4, 5], [6, 7]))
    print(x, x.shape)

    x = arraycomb(([1, 2, 3, 4]))
    print(x, x.shape)

    x = arraycomb([[0, 64, 128, 192, 256, 320, 384, 448], [0,  64, 128, 192, 256, 320, 384, 448]])
    print(x, x.shape)

    x1 = th.rand(5, 3, 4, 2)
    x2 = permute(x1, (0, 3, 1, 2))
    print(x1.shape)
    print(x2.shape)

    print(permute(x1, (1, 2), mode='matmul', dir='f').shape)
    print(permute(x2, (1, 2), mode='matmul', dir='b').shape)

    print(permute(x1, (0, 3, 1, 2), mode=None, dir='f').shape)
    print(permute(x2, (0, 3, 1, 2), mode=None, dir='b').shape)