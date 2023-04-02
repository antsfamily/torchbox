#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : dataset_visualization.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Dec 18 2022
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
from torchbox.utils.convert import str2num


def pca(x, dimn=0, ncmpnts='auto99%', algo='svd'):
    r"""Principal Component Analysis (pca) on raw data

    Parameters
    ----------
    x : tensor
        the input data
    dimn : int, optional
        the axis of number of samples, by default 0
    ncmpnts : int or str, optional
        the number of components, by default ``'auto99%'``
    algo : str, optional
        the kind of algorithms, by default ``'svd'``

    Returns
    -------
    tensor
        U, S, K (if :attr:`ncmpnts` is integer)

    Examples
    --------

    .. image:: ./_static/MNISTPCA_ORIG.png
       :scale: 100 %
       :align: center

    .. image:: ./_static/MNISTPCA_K70.png
       :scale: 100 %
       :align: center

    .. image:: ./_static/MNISTPCA_K90.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torch as th
        import torchbox as tb

        rootdir, dataset = '/mnt/d/DataSets/oi/dgi/mnist/official/', 'test'
        x, _ = tb.read_mnist(rootdir=rootdir, dataset=dataset, fmt='ubyte')
        print(x.shape)
        N, M2, _ = x.shape
        x = x.to(th.float32)

        u, s, k = tb.pca(x, dimn=0, ncmpnts='auto90%', algo='svd')
        print(u.shape, s.shape, k)
        u = u[:, :k]
        y = x.reshape(N, -1) @ u  # N-k
        z = y @ u.T.conj()
        z = z.reshape(N, M2, M2)
        print(tb.nmse(x, z, dim=(1, 2)))
        xp = th.nn.functional.pad(x[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
        zp = th.nn.functional.pad(z[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
        plt = tb.imshow(tb.patch2tensor(xp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Orignal'])
        plt = tb.imshow(tb.patch2tensor(zp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Reconstructed' + '(90%)'])

        u, s, k = tb.pca(x, dimn=0, ncmpnts='auto0.7', algo='svd')
        print(u.shape, s.shape, k)
        u = u[:, :k]
        y = x.reshape(N, -1) @ u  # N-k
        z = y @ u.T.conj()
        z = z.reshape(N, M2, M2)
        print(tb.nmse(x, z, dim=(1, 2)))
        zp = th.nn.functional.pad(z[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
        plt = tb.imshow(tb.patch2tensor(zp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Reconstructed' + '(70%)'])
        plt.show()

        u, s = tb.pca(x, dimn=0, ncmpnts=2, algo='svd')
        print(u.shape, s.shape)
        y = x.reshape(N, -1) @ u  # N-k
        z = y @ u.T.conj()
        z = z.reshape(N, M2, M2)
        print(tb.nmse(x, z, dim=(1, 2)))

    """


    xshape = x.shape
    N = xshape[dimn]
    if dimn != 0:
        x = x.transpose(0, dimn)
    x = th.reshape(x, (N, -1))
    N, M = x.shape
    
    x = x - th.mean(x, dim=1, keepdim=True)
    sigma = x.T.conj() @ x / M  # M-M

    if algo.lower() in ['svd']:
        [U, S, V] = th.linalg.svd(sigma)

    if type(ncmpnts) is int:
        return U[:, :ncmpnts], S[:ncmpnts]

    if ncmpnts.lower() in ['all']:
        return U, S

    if 'auto' in ncmpnts.lower():
        pct = str2num(ncmpnts, vfn=float)[0] / 100 if '%' in ncmpnts else str2num(ncmpnts, vfn=float)[0]
        lambd = th.abs(S)
        lambd_sum = th.sum(lambd)
        lambd_sumk, pctk = 0, 0
        K = 0
        for k in range(M):
            lambd_sumk += lambd[k]
            pctk = lambd_sumk / lambd_sum
            if pctk >= pct:
                K = k + 1
                break

        return U, S, K


if __name__ == '__main__':

    import torch as th
    import torchbox as tb

    rootdir, dataset = '/mnt/d/DataSets/oi/dgi/mnist/official/', 'test'
    x, _ = tb.read_mnist(rootdir=rootdir, dataset=dataset, fmt='ubyte')
    print(x.shape)
    N, M2, _ = x.shape
    x = x.to(th.float32)

    u, s, k = tb.pca(x, dimn=0, ncmpnts='auto90%', algo='svd')
    print(u.shape, s.shape, k)
    u = u[:, :k]
    y = x.reshape(N, -1) @ u  # N-k
    z = y @ u.T.conj()
    z = z.reshape(N, M2, M2)
    print(tb.nmse(x, z, dim=(1, 2)))
    xp = th.nn.functional.pad(x[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
    zp = th.nn.functional.pad(z[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
    plt = tb.imshow(tb.patch2tensor(xp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Orignal'])
    plt = tb.imshow(tb.patch2tensor(zp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Reconstructed' + '(90%)'])

    u, s, k = tb.pca(x, dimn=0, ncmpnts='auto0.7', algo='svd')
    print(u.shape, s.shape, k)
    u = u[:, :k]
    y = x.reshape(N, -1) @ u  # N-k
    z = y @ u.T.conj()
    z = z.reshape(N, M2, M2)
    print(tb.nmse(x, z, dim=(1, 2)))
    zp = th.nn.functional.pad(z[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
    plt = tb.imshow(tb.patch2tensor(zp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Reconstructed' + '(70%)'])
    plt.show()

    u, s = tb.pca(x, dimn=0, ncmpnts=2, algo='svd')
    print(u.shape, s.shape)
    y = x.reshape(N, -1) @ u  # N-k
    z = y @ u.T.conj()
    z = z.reshape(N, M2, M2)
    print(tb.nmse(x, z, dim=(1, 2)))
