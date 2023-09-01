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
import torchbox as tb


def pcat(x, sdim=-2, fdim=-1, isnorm=True, eigbkd='svd'):
    """gets Principal Component Analysis transformation

    Parameters
    ----------
    x : Tensor
        the input data
    sdim : int, optional
        the dimension index of sample, by default -2
    fdim : int or tuple, optional
        the dimension index of feature, by default -1
    isnorm : bool, optional
        whether to normalize covariance matric with the number of samples, by default True
    eigbkd : str, optional
        the backend of eigen decomposition, ``'svd'`` (default) or ``'eig'``

    Returns
    -------
    tensor
        the PCA transformation matrix, the eigenvalues

    """

    sdim = tb.dimpos(x.ndim, sdim)
    fdim = tb.dimpos(x.ndim, fdim)
    n = x.shape[sdim]

    mfdim, sdim = tb.dimmerge(x.ndim, mdim=fdim, dim=sdim, keepdim=False)
    x = tb.merge(x, dim=fdim, keepdim=False)

    x = tb.permute(x, dim=(mfdim, sdim), mode='matmul', dir='f')
    sigma = x @ (x.conj().transpose(-1, -2))
    
    if isnorm:
        sigma = sigma / n

    if eigbkd.lower() in ['svd']:
        U, S, _ = th.linalg.svd(sigma)
    if eigbkd.lower() in ['eig']:
        S, U = th.linalg.eig(sigma)

    if th.is_complex(x):
        return U, S
    else:
        return U.real, S.real

def pcapc(s, pcr=0.9):
    """get principal component according to the ratio of variance

    Parameters
    ----------
    s : Tensor
        eigenvalues
    pcr : float, optional
        the ratio of variance, by default 0.9
    """

    M = len(s)
    lambd = th.abs(s)
    lambd_sum = th.sum(lambd)
    lambd_sumk, pcrk = 0, 0
    K = 0
    for k in range(M):
        lambd_sumk += lambd[k]
        pcrk = lambd_sumk / lambd_sum
        if pcrk >= pcr:
            K = k + 1
            break

    return K

def pca(x, sdim=-2, fdim=-1, npcs='all', eigbkd='svd'):
    r"""Principal Component Analysis (pca) on raw data

    Parameters
    ----------
    x : Tensor
        the input data
    sdim : int, optional
        the dimension index of sample, by default -2
    fdim : int, optional
        the dimension index of feature, by default -1
    npcs : int or str, optional
        the number of components, by default ``'all'``
    eigbkd : str, optional
        the backend of eigen decomposition, ``'svd'`` (default) or ``'eig'``

    Returns
    -------
    tensor
        U, S, K (if :attr:`npcs` is integer)

    Examples
    --------

    .. image:: ./_static/MNISTPCA_ORIG.png
       :scale: 100 %
       :align: center

    .. image:: ./_static/MNISTPCA_K90.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        rootdir, dataset = '/mnt/d/DataSets/oi/dgi/mnist/official/', 'test'
        x, _ = tb.read_mnist(rootdir=rootdir, dataset=dataset, fmt='ubyte')
        print(x.shape)
        N, M2, _ = x.shape
        x = x.to(th.float32)
        pcr = 0.9

        u, s = tb.pca(x, sdim=0, fdim=(1, 2), eigbkd='svd')
        k = tb.pcapc(s, pcr=pcr)
        print(u.shape, s.shape, k)
        u = u[..., :k]
        y = x.reshape(N, -1) @ u  # N-k
        z = y @ u.T.conj()
        # z[z<0] = 0
        z = z.reshape(N, M2, M2)
        print(tb.nmse(x, z, dim=(1, 2)))
        xp = th.nn.functional.pad(x[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
        zp = th.nn.functional.pad(z[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
        plt = tb.imshow(tb.patch2tensor(xp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Orignal'])
        plt = tb.imshow(tb.patch2tensor(zp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Reconstructed with %d PCs(%.2f%%)' % (k, 100*k/u.shape[0])])
        plt.show()

        u, s = tb.pca(x.reshape(N, -1), sdim=0, fdim=1, npcs=2, eigbkd='svd')
        print(u.shape, s.shape)
        y = x.reshape(N, -1) @ u  # N-k
        z = y @ u.T.conj()
        z = z.reshape(N, M2, M2)
        print(tb.nmse(x, z, dim=(1, 2)))

    """

    x = x - th.mean(x, dim=fdim, keepdim=True)

    U, S = pcat(x, sdim=sdim, fdim=fdim, isnorm=True, eigbkd=eigbkd)

    if type(npcs) is int:
        return U[..., :npcs], S[..., :npcs]

    elif npcs.lower() in ['all']:
        return U, S
    else:
        raise ValueError('Not supported: %s!' % npcs)


if __name__ == '__main__':

    import torch as th
    import torchbox as tb

    rootdir, dataset = '/mnt/d/DataSets/oi/dgi/mnist/official/', 'test'
    x, _ = tb.read_mnist(rootdir=rootdir, dataset=dataset, fmt='ubyte')
    print(x.shape)
    N, M2, _ = x.shape
    x = x.to(th.float32)
    pcr = 0.9

    u, s = tb.pca(x, sdim=0, fdim=(1, 2), eigbkd='svd')
    k = tb.pcapc(s, pcr=pcr)
    print(u.shape, s.shape, k)
    u = u[..., :k]
    y = x.reshape(N, -1) @ u  # N-k
    z = y @ u.T.conj()
    # z[z<0] = 0
    z = z.reshape(N, M2, M2)
    print(tb.nmse(x, z, dim=(1, 2)))
    xp = th.nn.functional.pad(x[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
    zp = th.nn.functional.pad(z[:35], (1, 1, 1, 1, 0, 0), 'constant', 255)
    plt = tb.imshow(tb.patch2tensor(xp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Orignal'])
    plt = tb.imshow(tb.patch2tensor(zp, (5*(M2+2), 7*(M2+2)), dim=(1, 2)), titles=['Reconstructed with %d PCs(%.2f%%)' % (k, 100*k/u.shape[0])])
    plt.show()

    u, s = tb.pca(x.reshape(N, -1), sdim=0, fdim=1, npcs=2, eigbkd='svd')
    print(u.shape, s.shape)
    y = x.reshape(N, -1) @ u  # N-k
    z = y @ u.T.conj()
    z = z.reshape(N, M2, M2)
    print(tb.nmse(x, z, dim=(1, 2)))
