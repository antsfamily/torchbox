#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : mstar.py
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
import torch as th
import numpy as np


def mstar_header(filepath):
    r"""read header information of mstar file

    Parameters
    ----------
    filepath : str
        the mstar file path string.

    Returns
    -------
    dict
        header information dictionary.

    Examples
    --------

    The following example shows how to read the header information.

    ::

        import torchbox as tb

        datapath = tb.data_path('mstar')
        filepath = datapath + 'BTR70_HB03787.004'

        header = tb.mstar_header(filepath)
        for k, v in header.items():
            print(k, v)

    """

    f = open(filepath, 'rb')
    f.seek(1)
    header = {}
    version = f.readline().strip(b'\n').decode('utf-8')
    header['PhoenixHeaderVer'] = version[1+len('PhoenixHeaderVer'):-1]

    while True:
        data = f.readline().strip(b'\n').decode('utf-8')
        if data == '[EndofPhoenixHeader]':
            break
        if data:
            name, value = data.split('= ')
            header[name] = value
    f.close()

    return header


def mstar_raw(filepath, ofmt='c'):
    r"""load mstar raw data

    Each file is constructed with a prepended, variable-length, 
    Phoenix formatted (ASCII) header which contains detailed ground 
    truth and sensor information for the specific chip.  Following 
    the Phoenix header is the data block.  The data block is written 
    in Sun floating point format and is divided into two blocks, a 
    magnitude block followed by a phase block.  Byte swapping may be 
    required for certain host platforms.  Tools for reading and 
    manipulating the header information may be found at 
    https://www.sdms.afrl.af.mil .

    Parameters
    ----------
    filepath : str
        the data file path string.
    ofmt : str, optional
        output data type formation, ``'ap'`` for amplitude and angle,
        ``'c'`` for complex, and ``'r'`` for real and imaginary.

    Returns
    -------
    tensor
        the raw data with size :math:`{\mathbb C}^{H\times W}` (``'c'``), 
        :math:`{\mathbb R}^{H\times W \times 2}` (``'r'`` or ``'ap'``)

    Examples
    --------

    Read mstar raw amplitude-phase data and show in a figure.

    .. image:: ./_static/SHOW1_BTR70_HB03787.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torchbox as tb
        import matplotlib.pyplot as plt

        filepath = datapath + 'BTR70_HB03787.004'
        x = tb.mstar_raw(filepath, ofmt='ap')
        print(x.shape, th.max(x), th.min(x))

        plt.figure()
        plt.subplot(121)
        plt.imshow(x[..., 0])
        plt.title('amplitude')
        plt.subplot(122)
        plt.imshow(x[..., 1])
        plt.title('phase')
        plt.show()


    Read mstar raw complex-valued data and show in a figure.

    .. image:: ./_static/SHOW_BTR70_HB03787.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torchbox as tb
        import matplotlib.pyplot as plt

        filepath = datapath + 'BTR70_HB03787.004'
        x = tb.mstar_raw(filepath, ofmt='c')
        print(x.shape, th.max(x.abs()), th.min(x.abs()))

        plt.figure()
        plt.subplot(221)
        plt.imshow(x.real)
        plt.title('real part')
        plt.subplot(222)
        plt.imshow(x.imag)
        plt.title('imaginary part')
        plt.subplot(223)
        plt.imshow(x.abs())
        plt.title('amplitude')
        plt.subplot(224)
        plt.imshow(x.angle())
        plt.title('phase')
        plt.show()

    """

    f = open(filepath, 'rb')
    data = f.readline()
    data = f.readline()
    # PhoenixHeaderLength
    _, v = f.readline().strip(b'\n').decode('utf-8').split('= ')
    offsets = int(v)
    for k in range(9):
        f.readline()  # skip
    data, v = f.readline().strip(b'\n').decode('utf-8').split('= ')
    ncols = int(v)
    _, v = f.readline().strip(b'\n').decode('utf-8').split('= ')
    nrows = int(v)
    
    f.seek(offsets)
    # phase, angle
    x = struct.unpack_from('>%df' % (ncols*nrows*2), f.read(), 0)
    x = np.array(x).reshape(2, nrows, ncols)
    f.close()

    if ofmt in ['ap', 'amppha', 'AP', 'AMPPHA']:
        return th.from_numpy(np.transpose(x, (1, 2, 0)))
    if ofmt in ['r', 'real', 'REAL']:
        return th.from_numpy(np.stack((x[0, ...] * np.cos(x[1, ...]), x[0, ...] * np.sin(x[1, ...])), axis=-1))
    if ofmt in ['c', 'cplx', 'complex', 'C', 'CPLX', 'COMPLEX']:
        return th.from_numpy(x[0, ...] * (np.cos(x[1, ...]) + 1j * np.sin(x[1, ...])))

# def read_mstar(rootdir, dataset='test', fmt='bin'):
#     pass


if __name__ == '__main__':

    import torchbox as tb
    import matplotlib.pyplot as plt

    datapath = tb.data_path('mstar')
    filepath = datapath + 'BMP2_HB03787.000'
    filepath = datapath + 'BTR70_HB03787.004'
    # filepath = datapath + 'T72_HB03787.015'

    header = mstar_header(filepath)
    for k, v in header.items():
        print(k, v)

    x = mstar_raw(filepath, ofmt='ap')
    print(x.shape, th.max(x), th.min(x))

    plt.figure()
    plt.subplot(121)
    plt.imshow(x[..., 0])
    plt.title('amplitude')
    plt.subplot(122)
    plt.imshow(x[..., 1])
    plt.title('phase')
    plt.show()

    x = mstar_raw(filepath, ofmt='c')
    print(x.shape, th.max(x.abs()), th.min(x.abs()))

    plt.figure()
    plt.subplot(221)
    plt.imshow(x.real)
    plt.title('real part')
    plt.subplot(222)
    plt.imshow(x.imag)
    plt.title('imaginary part')
    plt.subplot(223)
    plt.imshow(x.abs())
    plt.title('amplitude')
    plt.subplot(224)
    plt.imshow(x.angle())
    plt.title('phase')
    plt.show()
