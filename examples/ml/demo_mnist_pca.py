#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : demo_mnist_pca.py
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
