#
# @file      : demo_read_fashionmnist.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Dec 11 2022
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

import torchbox as tb

# rootdir = '/mnt/d/DataSets/oi/dgi/mnist/pics/'
# dataset = 'test'
# X, Y = tb.read_mnist(rootdir=rootdir, dataset=dataset, fmt='image')
# print(X.shape, Y.shape)

rootdir = '/mnt/d/DataSets/oi/dgi/mnist/official/'
dataset = 'train'
X, Y = tb.read_mnist(rootdir=rootdir, dataset=dataset, fmt='ubyte')
print(X.shape, Y.shape)
plt = tb.imshow([X[i] for i in range(0, 32)])
plt.show()


dataset = 'test'
X, Y = tb.read_mnist(rootdir=rootdir, dataset=dataset, fmt='ubyte')
print(X.shape, Y.shape)

plt = tb.imshow([X[i] for i in range(0, 32)])
plt.show()
