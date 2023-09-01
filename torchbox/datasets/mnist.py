#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : mnist.py
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

import os
import struct
import numpy as np
import torch as th
from skimage.io import imread as skimread


def read_mnist(rootdir, dataset='test', fmt='ubyte'):
    r"""read mnist dataset

    The data can be downloaded from http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    rootdir : str
        root directory path string of mnist dataset 
    dataset : str, optional
        dataset to be read, ``'test'`` or ``'train'``, by default 'test'.
    fmt : str, optional
        the dataset formation, ``'ubyte'`` (original) or ``'image'`` (image), by default 'ubyte'.

    Returns
    -------
    list or tuple
        X : Tensor
            image data
        Y : Tensor
            label data
    
    Examples
    --------

    Read and show digital MNIST images

    .. image:: ./_static/mnist.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torchbox as tb

        rootdir = '/mnt/d/DataSets/oi/dgi/mnist/pics/'
        dataset = 'test'
        X, Y = tb.read_mnist(rootdir=rootdir, dataset=dataset, fmt='image')
        print(X.shape, Y.shape)

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

        # output
        (10000, 28, 28) (10000,)
        (60000, 28, 28) (60000,)
        (10000, 28, 28) (10000,)

    Read and show Fasion MNIST images

    .. image:: ./_static/fashionmnist.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torchbox as tb

        rootdir = '/mnt/d/DataSets/oi/dgi/fashionmnist/official/'
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

    """
    
    if fmt in ['ubyte', 'ubyte']:
        dataset = 't10k' if dataset == 'test' else dataset
        f = open(rootdir + '%s-images-idx3-ubyte' % dataset, 'rb')
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        X = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        f.close()
        f = open(rootdir + '%s-labels-idx1-ubyte' % dataset, 'rb')
        magic, num = struct.unpack('>II', f.read(8))
        Y = np.fromfile(f, dtype=np.uint8)
        f.close()

    if fmt in ['img', 'IMG']:
        X = []
        Y = []
        datasetpath = os.path.join(rootdir, dataset)
        if os.path.exists(datasetpath) is False:
            raise ValueError(datasetpath + " is not exist!")
        for n in range(10):
            for parent, dirnames, filenames in os.walk(datasetpath + '/' + str(n)):
                for filename in filenames:
                    img = skimread(parent + '/' + filename, 'L')  # RGB --> Gray
                    X.append(img)
                    Y.append(n)

        X = np.array(X)
        Y = np.array(Y)

    return th.from_numpy(X), th.from_numpy(Y)


if __name__ == '__main__':

    rootdir = '/mnt/d/DataSets/oi/dgi/mnist/pics/'
    dataset = 'test'
    X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='img')
    print(X.shape, Y.shape)

    rootdir = '/mnt/d/DataSets/oi/dgi/mnist/lecun/'
    dataset = 'train'
    X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='ubyte')
    print(X.shape, Y.shape)
    dataset = 'test'
    X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='ubyte')
    print(X.shape, Y.shape)
