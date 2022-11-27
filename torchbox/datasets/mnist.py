#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : mnist.py
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

import os
import struct
import numpy as np
import torch as th
from skimage.io import imread as skimread


def read_mnist(rootdir, dataset='test', fmt='bin'):
    """read mnist dataset

    The data can be downloaded from http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    rootdir : str
        root directory path string of mnist dataset 
    dataset : str, optional
        dataset to be read, ``'test'`` or ``'train'``, by default 'test'.
    fmt : str, optional
        the dataset formation, ``'bin'`` (original) or ``'img'`` (image), by default 'bin'.

    Returns
    -------
    list or tuple
        X : tensor
            image data
        Y : tensor
            label data
    
    Examples
    --------

        ::

            rootdir = '/mnt/d/DataSets/oi/dgi/mnist/pics/'
            dataset = 'test'
            X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='img')
            print(X.shape, Y.shape)

            rootdir = '/mnt/d/DataSets/oi/dgi/mnist/lecun/'
            dataset = 'train'
            X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='bin')
            print(X.shape, Y.shape)
            dataset = 'test'
            X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='bin')
            print(X.shape, Y.shape)

            # output
            (10000, 28, 28) (10000,)
            (60000, 28, 28) (60000,)
            (10000, 28, 28) (10000,)
    """
    
    if fmt in ['bin', 'BIN']:
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
    X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='bin')
    print(X.shape, Y.shape)
    dataset = 'test'
    X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='bin')
    print(X.shape, Y.shape)
