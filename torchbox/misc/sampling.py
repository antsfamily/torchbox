#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : sampling.py
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
import numpy as np
from torchbox.base.arrayops import sl
from torchbox.base.randomfunc import setseed, randgrid, randperm
from torchbox.base.arrayops import arraycomb
from torchbox.utils.ios import loadmat, loadh5


def slidegrid(start, stop, step, shake=0, n=None):
    r"""generates sliding grid indexes

    Generates :attr:`n` sliding grid indexes from :attr:`start` to :attr:`stop`
    with step size :attr:`step`.

    Args:
        start (int or list): start sampling point
        stop (int or list): stop sampling point
        step (int or list): sampling stepsize
        shake (float): the shake rate, if :attr:`shake` is 0, no shake, (default),
            if positive, add a positive shake, if negative, add a negative.
        n (int or None): the number of samples (default None, int((stop0 - start0) / step0) * int((stop1 - start1) / step1)...).

    Returns:
        for multi-dimension, return a 2-d tensor, for 1-dimension, return a 1d-tensor.

    Raises:
        TypeError: The number of samples should be an integer or None.

    see also :func:`randperm`, :func:`randgrid`.

    """

    starts = [start] if type(start) is int else start
    stops = [stop] if type(stop) is int else stop
    steps = [step] if type(step) is int else step
    shakes = [shake] if type(shake) is int or type(shake) is float else shake
    if (n is not None) and (type(n) is not int):
        raise TypeError('The number of samples should be an integer or None!')
    elif n is None:
        n = float('inf')
    index = []
    for start, stop, step, shake in zip(starts, stops, steps, shakes):
        shakep = shake if abs(shake) >= 1 and type(shake) is int else int(shake * step)
        x = th.tensor(range(start, stop, step))
        if shakep != 0:
            s = th.randint(0, abs(shakep) + 1, (len(x),))
            x = x - s if shakep < 0 else x + s
            x[x >= (stop - step)] = stop - step
            x[x < start] = start
        index.append(x)
    P = arraycomb(index)
    n = min(P.shape[0], n)
    P = P[:n, ...]

    if len(starts) == 1:
        P = P.squeeze(1)
        return P
    else:
        return P.t()


def dnsampling(x, ratio=1., dim=-1, smode='uniform', omode='discard', seed=None, retall=False):
    """Summary

    Args:
        x (Tensor): The Input tensor.
        ratio (float, optional): Downsampling ratio.
        dim (int, optional): Downsampling axis (default -1).
        smode (str, optional): Downsampling mode: ``'uniform'``, ``'random'``, ``'random2'``.
        omode (str, optional): output mode: ``'discard'`` for discarding, ``'zero'`` for zero filling.
        seed (int or None, optional): seed for torch's random.
        retall (bool, optional): If ``True``, also return sampling mask.

    Returns:
        (Tensor): downsampled data.

    Raises:
        TypeError: :attr:`axis`
        ValueError: :attr:`ratio`, attr:`smode`, attr:`omode`
    """

    axis = dim
    nDims = x.dim()
    if type(axis) is int:
        if type(ratio) is not float:
            raise ValueError('Downsampling ratio should be a number!')
        axis = [axis]
        ratio = [ratio]
    elif type(axis) is list or tuple:
        if len(axis) != len(ratio):
            raise ValueError('You should specify the DS ratio for each axis!')
    else:
        raise TypeError('Wrong type of axis!')

    axis, ratio = list(axis), list(ratio)
    for cnt in range(len(axis)):
        if axis[cnt] < 0:
            axis[cnt] += nDims
        # ratio[cnt] = 1. - ratio[cnt]
        cnt += 1

    if omode in ['discard', 'DISCARD', 'Discard']:
        if smode not in ['uniform', 'UNIFORM', 'Uniform']:
            raise ValueError("Only support uniform mode!")

        index = [slice(None)] * nDims
        for a, r in zip(axis, ratio):
            sa = x.shape[a]
            da = int(round(1. / r))
            index[a] = slice(0, sa, da)
        index = tuple(index)

        if retall:
            return x[index], index
        else:
            return x[index]

    elif omode in ['zero', 'ZERO', 'Zeros']:
        mshape = [1] * nDims
        for a in axis:
            mshape[a] = x.shape[a]
        mask = th.zeros(mshape, dtype=th.uint8, device=x.device)
        if smode in ['uniform', 'UNIFORM', 'Uniform']:
            for a, r in zip(axis, ratio):
                sa = x.shape[a]
                da = int(round(1. / r))
                idx = sl(nDims, a, slice(0, sa, da))
                mask[idx] += 1
            mask[mask < len(axis)] = 0
            mask[mask >= len(axis)] = 1

        elif smode in ['random', 'RANDOM', 'Random']:
            setseed(seed, target='torch')
            for a, r in zip(axis, ratio):
                d = x.dim()
                s = x.shape[a]
                n = int(round(s * r))
                idx = randperm(0, s, n)
                idx = np.sort(idx)
                idx = sl(d, a, idx)
                mask[idx] += 1
            mask[mask < len(axis)] = 0
            mask[mask >= len(axis)] = 1

        elif smode in ['random2', 'RANDOM2', 'Random2']:
            setseed(seed, target='torch')
            d = x.dim()
            s0, s1 = x.shape[axis[0]], x.shape[axis[1]]
            n0, n1 = int(round(s0 * ratio[0])), int(round(s1 * ratio[0]))
            idx0 = randperm(0, s0, n0)
            # idx0 = np.sort(idx0)

            for i0 in idx0:
                idx1 = randperm(0, s1, n1)
                mask[sl(d, [axis[0], axis[1]], [[i0], idx1])] = 1

        else:
            raise ValueError('Not supported sampling mode: %s!' % smode)

        if retall:
            return x * mask, mask
        else:
            return x * mask

    else:
        raise ValueError('Not supported output mode: %s!' % omode)


def sample_tensor(x, n, dim=0, groups=1, mode='sequentially', seed=None, retall=False):
    r"""sample a tensor

    Sample a tensor sequentially/uniformly/randomly.

    Args:
        x (torch.Tensor): a torch tensor to be sampled
        n (int): sample number
        dim (int, optional): the axis to be sampled (the default is 0)
        groups (int, optional): number of groups in this tensor (the default is 1)
        mode (str, optional): - ``'sequentially'``: evenly spaced (default)
            - ``'uniformly'``: [0, int(n/groups)]
            - ``'randomly'``: randomly selected, non-returned sampling
        seed (None or int, optional): only work for ``'randomly'`` mode (the default is None)
        retall (bool, optional): If ``True``, also return the selected indexes, the default is ``False``.

    Returns:
        y (torch.Tensor): Sampled torch tensor.
        idx (list): Sampled indexes, if :attr:`retall` is ``True``, this will also be returned.


    Example:
    
        ::

            setseed(2020, 'torch')

            x = th.randint(1000, (20, 3, 4))
            y1, idx1 = sample_tensor(x, 10, axis=0, groups=2, mode='sequentially', retall=True)
            y2, idx2 = sample_tensor(x, 10, axis=0, groups=2, mode='uniformly', retall=True)
            y3, idx3 = sample_tensor(x, 10, axis=0, groups=2, mode='randomly', retall=True)

            print(x.shape)
            print(y1.shape)
            print(y2.shape)
            print(y3.shape)
            print(idx1)
            print(idx2)
            print(idx3)

            the outputs are as follows:

            torch.Size([20, 3, 4])
            torch.Size([10, 3, 4])
            torch.Size([10, 3, 4])
            torch.Size([10, 3, 4])
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
            [3, 1, 5, 8, 7, 17, 18, 13, 16, 10]


    Raises:
        ValueError: The tensor does not has enough samples.


    """

    axis = dim
    N = x.shape[axis]
    M = int(N / groups)  # each group has M samples
    m = int(n / groups)  # each group has m sampled samples

    if (M < m):
        raise ValueError('The tensor does not has enough samples')

    idx = []
    if mode in ['sequentially', 'Sequentially']:
        for g in range(groups):
            idx += list(range(int(M * g), int(M * g) + m))
    if mode in ['uniformly', 'Uniformly']:
        for g in range(groups):
            idx += list(range(int(M * g), int(M * g + M), int(M / m)))[:m]
    if mode in ['randomly', 'Randomly']:
        setseed(seed, target='torch')
        for g in range(groups):
            idx += list(randperm(int(M * g), int(M * g + M), m).numpy())

    if retall:
        return x[sl(x.dim(), axis=axis, idx=[idx])], idx
    else:
        return x[sl(x.dim(), axis=axis, idx=[idx])]


def shuffle_tensor(x, dim=0, groups=1, mode='inter', seed=None, retall=False):
    """shuffle a tensor

    Shuffle a tensor randomly.

    Args:
        x (Tensor): A torch tensor to be shuffled.
        dim (int, optional): The axis to be shuffled (default 0)
        groups (number, optional): The number of groups in this tensor (default 1)
        mode (str, optional):
            - ``'inter'``: between groups (default)
            - ``'intra'``: within group
            - ``'whole'``: the whole
        seed (None or number, optional): random seed (the default is None)
        retall (bool, optional): If ``True``, also returns the shuffle indexes, the default is ``False``.

    Returns:
        y (Tensor): Shuffled torch tensor.
        idx (list): Shuffled indexes, if :attr:`retall` is ``True``, this will also be returned.


    Example:

        ::

            setseed(2020, 'torch')

            x = th.randint(1000, (20, 3, 4))
            y1, idx1 = shuffle_tensor(x, axis=0, groups=4, mode='intra', retall=True)
            y2, idx2 = shuffle_tensor(x, axis=0, groups=4, mode='inter', retall=True)
            y3, idx3 = shuffle_tensor(x, axis=0, groups=4, mode='whole', retall=True)

            print(x.shape)
            print(y1.shape)
            print(y2.shape)
            print(y3.shape)
            print(idx1)
            print(idx2)
            print(idx3)

            the outputs are as follows:

            torch.Size([20, 3, 4])
            torch.Size([20, 3, 4])
            torch.Size([20, 3, 4])
            torch.Size([20, 3, 4])
            [1, 0, 3, 4, 2, 8, 6, 5, 9, 7, 13, 11, 12, 14, 10, 18, 15, 17, 16, 19]
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
            [1, 13, 12, 5, 19, 9, 11, 6, 4, 16, 17, 3, 8, 18, 7, 10, 15, 0, 14, 2]


    """

    axis = dim
    N = x.shape[axis]
    M = int(N / groups)  # each group has M samples

    idx = []
    setseed(seed, target='torch')
    if mode in ['whole', 'Whole', 'WHOLE']:
        idx = list(randperm(0, N, N).numpy())

    if mode in ['intra', 'Intra', 'INTRA']:
        for g in range(groups):
            idx += list(randperm(int(M * g), int(M * g + M), M).numpy())
    if mode in ['inter', 'Inter', 'INTER']:
        for g in range(groups):
            idx += [list(range(int(M * g), int(M * g + M)))]
        groupidx = list(randperm(0, groups, groups).numpy())

        iidx = idx.copy()
        idx = []
        for i in groupidx:
            idx += iidx[i]

    if retall:
        return x[sl(x.dim(), axis=axis, idx=[idx])], idx
    else:
        return x[sl(x.dim(), axis=axis, idx=[idx])]


def split_tensor(x, ratios=[0.7, 0.2, 0.1], dim=0, shuffle=False, seed=None, retall=False):
    """split tensor

    split a tensor into some parts.

    Args:
        x (Tensor): A torch tensor.
        ratios (list, optional): Split ratios (the default is [0.7, 0.2, 0.05])
        dim (int, optional): Split axis (the default is 0)
        shuffle (bool, optional): Whether shuffle (the default is False)
        seed (int, optional): Shuffule seed (the default is None)
        retall (bool, optional): If ``True``, also return the split indexes, the default is ``False``.

    Returns:
        (list of Tensor): Splitted tensors.
    """

    y, idxes = [], []

    axis = dim
    N, ns = x.shape[axis], 0
    if shuffle:
        setseed(seed, target='torch')
        idx = randperm(0, N, N)
    else:
        idx = list(range(N))

    for ratio in ratios:
        n = int(ratio * N)
        idxes.append(idx[ns:ns + n])
        y.append(x[sl(x.dim(), axis=axis, idx=[idx[ns:ns + n]])])
        ns += n

    if retall:
        return y, idxes
    else:
        return y


def cut_tensor(x, start=0, stop=-1, dim=0, groups=1, retall=False):
    """cut tensor

    cut a part of tensor.

    Args:
        x (Tensor): A torch tensor.
        start (int, optional): start position (the default is 0)
        stop (int, optional): stop position (the default is -1)
        dim (int, optional): cut axis (the default is 0)
        groups (number, optional): The number of groups in this tensor (default 1)
        retall (bool, optional): If ``True``, also return the cut indexes, the default is ``False``.

    Returns:
        (list of Tensor): cutted tensor.
    
    """

    N = x.shape[dim]
    M = int(N / groups)  # each group has M samples

    if stop == -1:
        stop = M
    
    if (M < start) or (M < stop):
        raise ValueError('The tensor does not has enough samples')

    idx = []
    for g in range(groups):
        idx += list(range(int(M * g) + start, int(M * g) + stop))

    if retall:
        return x[sl(x.dim(), axis=dim, idx=[idx])], idx
    else:
        return x[sl(x.dim(), axis=dim, idx=[idx])]


def tensor2patch(x, n=None, size=(32, 32), dim=(0, 1), start=(0, 0), stop=(None, None), step=(1, 1), shake=(0, 0), mode='slidegrid', seed=None):
    """sample patch from a tensor

    Sample some patches from a tensor, tensor and patch can be any size.

    Args:
        x (Tensor): Tensor to be sampled.
        n (int, optional): The number of pactches, the default is None, auto computed,
            equals to the number of blocks with specified :attr:`step`
        size (tuple or int, optional): The size of patch (the default is (32, 32))
        dim (tuple or int, optional): The sampling axis (the default is (0, 1))
        start (tuple or int, optional): Start sampling index for each axis (the default is (0, 0))
        stop (tuple or int, optional): Stopp sampling index for each axis. (the default is (None, None), which [default_description])
        step (tuple or int, optional): Sampling stepsize for each axis  (the default is (1, 1), which [default_description])
        shake (tuple or int or float, optional): float for shake rate, int for shake points (the default is (0, 0), which means no shake)
        mode (str, optional): Sampling mode, ``'slidegrid'``, ``'randgrid'``, ``'randperm'`` (the default is 'slidegrid')
        seed (int, optional): Random seed. (the default is None, which means no seed.)

    Returns:
        (Tensor): A Tensor of sampled patches.
    """

    axis = dim
    axis = [axis] if type(axis) is int else list(axis)
    naxis = len(axis)
    sizep = [size] * naxis if type(size) is int else list(size)
    start = [start] * naxis if type(start) is int else list(start)
    stop = [stop] * naxis if type(stop) is int else list(stop)
    step = [step] * naxis if type(step) is int else list(step)
    shake = [shake] * naxis if type(shake) is float else list(shake)

    dimx = x.dim()
    dimp = len(axis)
    sizex = np.array(x.shape)
    sizep = np.array(sizep)

    npatch = []
    npatch = np.uint32(sizex[axis] - sizep) // step + 1
    N = int(np.prod(npatch))
    n = N if n is None else int(n)

    yshape = list(x.shape)
    for a, p in zip(axis, sizep):
        yshape[a] = p
    yshape = [n] + yshape

    for i in range(naxis):
        if stop[i] is None:
            stop[i] = sizex[axis[i]]

    stop = stop - sizep + 1

    y = th.zeros(yshape, dtype=x.dtype, device=x.device)

    if mode in ['slidegrid', 'SLIDEGRID', 'SlideGrid']:
        assert n <= N, ('n should be slower than ' + str(N + 1))
        seppos = slidegrid(start, stop, step, shake, n)
    if mode in ['randgrid', 'RANDGRID', 'RandGrid']:
        assert n <= N, ('n should be slower than ' + str(N + 1))
        setseed(seed, target='torch')
        seppos = randgrid(start, stop, step, shake, n)

    if mode in ['randperm', 'RANDPERM', 'RandPerm']:
        setseed(seed, target='torch')
        seppos = randgrid(start, stop, [1] * dimp, [0] * dimp, n)

    for i in range(n):
        indexi = []
        for j in range(dimp):
            indexi.append(slice(seppos[j][i].item(), seppos[j][i].item() + sizep[j]))
        t = x[sl(dimx, axis, indexi)]
        y[i] = t
    return y


def patch2tensor(p, size=(256, 256), dim=(1, 2), start=(0, 0), stop=(None, None), step=None, mode='nfirst'):
    """merge patch to a tensor


    Args:
        p (Tensor): A tensor of patches.
        size (tuple, optional): Merged tensor size in the dimension (the default is (256, 256)).
        dim (tuple, optional): Merged axis of patch (the default is (1, 2))
        start (tuple, optional): start position for placing patch (the default is (0, 0))
        stop (tuple, optional): stop position for placing patch (the default is (0, 0))
        step (tuple, optional): step size for placing patch (the default is :obj:`None`, which means the size of patch)
        mode (str, optional): Patch mode ``'nfirst'`` or ``'nlast'`` (the default is 'nfirst',
            which means the first dimension is the number of patches)

    Returns:
        Tensor: Merged tensor.
    """

    axis = dim
    axis = list(axis)
    naxis = len(axis)
    sizep = np.array(p.shape)
    sizex = np.array(p.shape)
    sizex[axis] = size
    
    dimp = p.dim()
    axisp = np.array(range(0, dimp))
    steps = np.copy(sizep[axis]) if step is None else step
    npatch = np.uint32(sizex[axis] - sizep[axis]) // steps + 1

    xaxis = np.copy(axis)
    sizex2 = sizex
    if mode in ['nfirst', 'Nfirst', 'NFIRST']:
        axisn = 0
        N = p.shape[0]
        sizex2 = sizex[1:]
        xaxis = [a - 1 for a in axis]

    if mode in ['nlast', 'Nlast', 'NLAST']:
        axisn = -1
        N = p.shape[-1]
        sizex2 = sizex[:-1]

    x = th.zeros(list(sizex2), dtype=p.dtype, device=p.device)

    dimx = x.dim()
    axisx = list(range(dimx))

    index = []
    for stop, size, step in zip(sizex[axis], sizep[axis], steps):
        idx = np.array(range(0, stop - size + 1, step))
        index.append(idx)
    index = arraycomb(index)
    for n in range(N):
        indexn = []
        for a in xaxis:
            indexn.append(slice(index[n, a].item(), index[n, a].item() + steps[a], 1))
        x[sl(dimx, xaxis, indexn)] = p[sl(dimp, axisn, n)]
    return x


def read_samples(datafiles, keys=[['SI', 'ca', 'cr']], nsamples=[10], groups=[1], mode='sequentially', dim=0, parts=None, seed=None):
    """read samples from files

    Parameters
    ----------
    datafiles : list
        list of path strings
    keys : list, optional
        data keys for reading, by default [['SI', 'ca', 'cr']]
    nsamples : list, optional
        number of samples for each data file, by default [10]
    groups : list, optional
        number of groups in each data file, by default [1]
    mode : str, optional
        sampling mode for all datafiles, by default 'sequentially'
    dim : int, optional
        sampling dimension/axis for all datafiles, by default 0
    parts : int or None, optional
        number of parts (split samples into some parts), by default None
    seed : int or None, optional
        the seed for random stream, by default None

    Returns:
        tensor: samples

    Raises:
        ValueError: :attr:`nsamples` should be large enough
    """

    axis = dim
    nfiles = len(datafiles)
    if len(keys) == 1:
        keys = keys * nfiles
    if len(nsamples) == 1:
        nsamples = nsamples * nfiles
    if len(groups) == 1:
        groups = groups * nfiles

    nkeys = len(keys[0])

    if parts is None:
        outs = [th.tensor([])] * nkeys
    else:
        nparts = len(parts)
        outs = [[th.tensor([])] * nparts] * nkeys

    for datafile, key, n, group in zip(datafiles, keys, nsamples, groups):

        if datafile[datafile.rfind('.'):] == '.mat':
            data = loadmat(datafile)
        if datafile[datafile.rfind('.'):] in ['.h5', '.hdf5']:
            data = loadh5(datafile)

        N = data[key[0]].shape[axis]
        M = int(N / group)  # each group has M samples
        m = int(n / group)  # each group has m sampled samples

        if (M < m):
            raise ValueError('The tensor does not has enough samples')

        idx = []
        if mode in ['sequentially', 'Sequentially']:
            for g in range(group):
                idx += list(range(int(M * g), int(M * g) + m))
        if mode in ['uniformly', 'Uniformly']:
            for g in range(group):
                idx += list(range(int(M * g), int(M * g + M), int(M / m)))[:m]
        if mode in ['randomly', 'Randomly']:
            setseed(seed)
            for g in range(group):
                idx += randperm(int(M * g), int(M * g + M), m)

        for j, k in enumerate(key):
            d = np.ndim(data[k])
            if parts is None:
                outs[j] = th.cat((outs[j], th.from_numpy(data[k][sl(d, axis, [idx])])), axis=axis)
            else:
                nps, npe = 0, 0
                for i in range(nparts):
                    part = parts[i]
                    npe = nps + int(part * group)
                    outs[j][i] = th.cat((outs[j][i], th.from_numpy(data[k][sl(d, axis, [idx[nps:npe]])])), axis=axis)
                    nps = npe

    return outs


if __name__ == '__main__':

    setseed(2020, 'torch')
    x = th.randint(1000, (20, 3, 4))
    y1, idx1 = sample_tensor(x, 10, dim=0, groups=2, mode='sequentially', retall=True)
    y2, idx2 = sample_tensor(x, 10, dim=0, groups=2, mode='uniformly', retall=True)
    y3, idx3 = sample_tensor(x, 10, dim=0, groups=2, mode='randomly', retall=True)

    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    print(idx1)
    print(idx2)
    print(idx3)

    x = th.randint(1000, (20, 3, 4))
    y1, idx1 = shuffle_tensor(x, dim=0, groups=4, mode='intra', retall=True)
    y2, idx2 = shuffle_tensor(x, dim=0, groups=4, mode='inter', retall=True)
    y3, idx3 = shuffle_tensor(x, dim=0, groups=4, mode='whole', retall=True)

    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    print(idx1)
    print(idx2)
    print(idx3)

    y1, y2, y3 = split_tensor(x, ratios=[0.7, 0.2, 0.1], dim=0, shuffle=False, seed=None)
    print(y3)
    y1, y2, y3 = split_tensor(x, ratios=[0.7, 0.2, 0.1], dim=0, shuffle=True, seed=None)
    print(y3)
    y1, y2, y3 = split_tensor(x, ratios=[0.7, 0.2, 0.1], dim=0, shuffle=True, seed=2021)
    print(y3)
    y1, y2, y3 = split_tensor(x, ratios=[0.7, 0.2, 0.1], dim=0, shuffle=True, seed=2021)
    print(y3)
    print(y1.shape, y2.shape, y3.shape)

    x = th.randint(1000, (20, 3, 4))
    y1, idx1 = cut_tensor(x, start=2, stop=6, dim=0, groups=2, retall=True)

    print(x.shape)
    print(y1.shape)
    print(idx1)

    Na, Nr, Nc = (9, 12, 2)
    x = th.randint(1000, (Na, Nr, Nc))

    print(x[:, :, 0], 'x', x.shape)
    print(x[:, :, 1], 'x', x.shape)

    y = dnsampling(x, ratio=(0.5, 0.5), dim=(0, 1), smode='uniform', omode='discard')
    print(y[:, :, 0], 'discard')
    print(y[:, :, 1], 'discard')

    y = dnsampling(x, ratio=(0.5, 0.5), dim=(0, 1), smode='uniform', omode='zero')
    print(y[:, :, 0], 'zero')
    print(y[:, :, 1], 'zero')

    y = dnsampling(x, ratio=(0.5, 0.5), dim=(0, 1), smode='random', omode='zero')
    print(y[:, :, 0], 'zero')
    print(y[:, :, 1], 'zero')

    y = dnsampling(x, ratio=(0.5, 0.5), dim=(0, 1), smode='random2', omode='zero')
    print(y[:, :, 0], 'zero')
    print(y[:, :, 1], 'zero')

    # y = tensor2patch(x, n=None, size=(2, 3), dim=(0, 1), mode='slide', step=(1, 1), seed=None)
    # print(y.shape, 'slide')
    # print(y[0, :, :, 0], 'slide')
    # print(y[0, :, :, 1], 'slide')
    y = tensor2patch(x, n=None, size=(2, 3), dim=(0, 1), step=(2, 3), shake=(0, 0), mode='randgrid', seed=None)
    print(y.shape, 'randgrid')
    print(y[0, :, :, 0], 'randgrid')
    print(y[0, :, :, 1], 'randgrid')

    y = tensor2patch(x, n=None, size=(2, 3), dim=(0, 1), step=(1, 1), shake=(0, 0), mode='randperm', seed=None)
    print(y.shape, 'randperm')
    print(y[0, :, :, 0], 'randperm')
    print(y[0, :, :, 1], 'randperm')
