#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : mathops.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 1.0
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

import scipy
import numpy as np
import torch as th
import torchbox as tb
from torch.autograd import Function


def svd_rank(A, svdr='auto'):
    r"""compute rank of the truncated Singular Value Decomposition

    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.

    Parameters
    ----------
    A : Tensor
        The input matrix
    svdr : str or int, optional
        the rank for the truncation, ``'auto'`` for automatic computation, by default ``'auto'``
    """

    U, s, _ = th.linalg.svd(A, full_matrices=False)

    if svdr.lower() in ['a', 'auto']:
        beta = A.shape[0] / (1.0*A.shape[1]) if A.shape[0] < A.shape[1] else A.shape[1] / (1.0*A.shape[0])
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau = th.median(s) * omega
        rank = th.sum(s > tau)
        rank = 1 if rank == 0 else rank
    elif (type(svdr) is int) and (svdr >= 1):
        rank = min(svdr, U.shape[-1])
    elif 0 < svdr < 1:
        cum_energy = th.cumsum(s**2 / (s**2).sum())
        rank = th.searchsorted(cum_energy, svdr) + 1
    else:
        raise ValueError('Not supported svdr mode!')
    
    return rank

def eig(A, cdim=None, dim=(-2, -1), keepdim=False):
    r"""Computes the eigenvalues and eigenvectors of a square matrix.

    Parameters
    ----------
    A : Tensor
        any size tensor, both complex and real representation are supported.
        For real representation, the real and imaginary dimension is specified by :attr:`cdim` or :attr:`caxis`.
    cdim : int or None, optional
        if :attr:`A` and :attr:`B` are complex tensors but represented in real format, :attr:`cdim` or :attr:`caxis`
        should be specified (Default is :obj:`None`).
    dim : tulpe or list
        dimensions for multiplication (default is (-2, -1))
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    """

    if th.is_complex(A):
        A = tb.permute(A, dim=dim, mode='matmul', dir='f')
        return th.linalg.eig(A)
    elif cdim is None:
        A = tb.permute(A, dim=dim, mode='matmul', dir='f')
        return th.linalg.eig(A)
    else:
        dim = tb.rmcdim(A.ndim, dim=dim, cdim=cdim, keepdim=keepdim)
        A = tb.r2c(A, cdim=cdim, keepdim=keepdim)
        A = tb.permute(A, dim=dim, mode='matmul', dir='f')
        return th.linalg.eig(A)

def eigvals(A, cdim=None, dim=(-2, -1), keepdim=False):
    """Computes the eigenvalues of a square matrix.

    Parameters
    ----------
    A : Tensor
        any size tensor, both complex and real representation are supported.
        For real representation, the real and imaginary dimension is specified by :attr:`cdim` or :attr:`caxis`.
    cdim : int or None, optional
        if :attr:`A` and :attr:`B` are complex tensors but represented in real format, :attr:`cdim` or :attr:`caxis`
        should be specified (Default is :obj:`None`).
    dim : tulpe or list
        dimensions for multiplication (default is (-2, -1))
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    """

    if th.is_complex(A):
        A = tb.permute(A, dim=dim, mode='matmul', dir='f')
        return th.linalg.eigvals(A)
    elif cdim is None:
        A = tb.permute(A, dim=dim, mode='matmul', dir='f')
        return th.linalg.eigvals(A)
    else:
        dim = tb.rmcdim(A.ndim, dim=dim, cdim=cdim, keepdim=keepdim)
        A = tb.r2c(A, cdim=cdim, keepdim=keepdim)
        A = tb.permute(A, dim=dim, mode='matmul', dir='f')
        return th.linalg.eigvals(A)

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = th.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = th.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply