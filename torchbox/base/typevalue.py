#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import torch as th
from torchbox import c2r, nextpow2, str2num


def dtypes(t='int'):
    if t in ['int', 'INT', 'Int']:
        return [th.int, th.int8, th.int16, th.int32, th.int64]
    if t in ['uint', 'UINT', 'UInt']:
        return [th.uint8]
    if t in ['float', 'FLOAT', 'Float']:
        return [th.float, th.float16, th.float32, th.float64]
    if t in ['complex', 'COMPLEX', 'Complex']:
        return [th.complex, th.complex64, th.complex128]


def peakvalue(A):
    r"""Compute the peak value of the input.

    Find peak value in matrix

    Parameters
    ----------
    A : numpy array
        Data for finding peak value

    Returns
    -------
    number
        Peak value.
    """

    if th.is_complex(A):  # complex in complex
        A = c2r(A)

    dtype = A.dtype
    if dtype in dtypes('float'):
        maxv = th.max(A)
        Vpeak = 1 if maxv < 1 else 2**nextpow2(maxv) - 1
    elif dtype == dtypes('uint'):
        datatype = str(dtype)
        Vpeak = 2 ** str2num(datatype, int)[0] - 1
    elif dtype in dtypes('int'):
        datatype = str(dtype)
        Vpeak = 2 ** str2num(datatype, int)[0] / 2 - 1
    else:
        print("~~~Unknown type: %s, using the maximum value!" % dtype)
        Vpeak = th.max(A.abs())

    return Vpeak


if __name__ == '__main__':
    
    x = th.rand(3, 4)
    print(x.dtype, type(x.dtype))

    peakvalue(x)

