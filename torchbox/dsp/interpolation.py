#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : interpolation.py
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

import torch.nn.functional as thf


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    r"""Down/up samples the input to either the given :attr:`size` or the given
        :attr:`scale_factor`

        The algorithm used for interpolation is determined by :attr:`mode`.

        Currently temporal, spatial and volumetric sampling are supported, i.e.
        expected inputs are 3-D, 4-D or 5-D in shape.

        The input dimensions are interpreted in the form:
        `mini-batch x channels x [optional depth] x [optional height] x width`.

        The modes available for resizing are: `nearest`, `linear` (3D-only),
        `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

        Args:
            input (Tensor): the input tensor
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
                output spatial size.
            scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
            mode (str): algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
            align_corners (bool, optional): Geometrically, we consider the pixels of the
                input and output as squares rather than points.
                If set to ``True``, the input and output tensors are aligned by the
                center points of their corner pixels, preserving the values at the corner pixels.
                If set to ``False``, the input and output tensors are aligned by the corner
                points of their corner pixels, and the interpolation uses edge value padding
                for out-of-boundary values, making this operation *independent* of input size
                when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
                is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
                Default: ``False``
            recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
                interpolation calculation.  When `scale_factor` is passed as a parameter, it is used
                to compute the `output_size`.  If `recompute_scale_factor` is ```False`` or not specified,
                the passed-in `scale_factor` will be used in the interpolation computation.
                Otherwise, a new `scale_factor` will be computed based on the output and input sizes for
                use in the interpolation computation (i.e. the computation will be identical to if the computed
                `output_size` were passed-in explicitly).  Note that when `scale_factor` is floating-point,
                the recomputed scale_factor may differ from the one passed in due to rounding and precision
                issues.

        .. note::
            With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
            negative values or values greater than 255 for images.
            Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
            when displaying the image.

        .. warning::
            With ``align_corners = True``, the linearly interpolating modes
            (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
            output and input pixels, and thus the output values can depend on the
            input size. This was the default behavior for these modes up to version
            0.3.1. Since then, the default behavior is ``align_corners = False``.
            See :class:`~th.nn.Upsample` for concrete examples on how this
            affects the outputs.

        .. warning::
            When scale_factor is specified, if recompute_scale_factor=True,
            scale_factor is used to compute the output_size which will then
            be used to infer new scales for the interpolation.
            The default behavior for recompute_scale_factor changed to False
            in 1.6.0, and scale_factor is used in the interpolation
            calculation.

        Note:
            When using the CUDA backend, this operation may induce nondeterministic
            behaviour in its backward pass that is not easily switched off.
            Please see the notes on :doc:`/notes/randomness` for background.
        """

    return thf.interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor)


def interpolatec(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    r"""Down/up samples the input to either the given :attr:`size` or the given
        :attr:`scale_factor`

        The algorithm used for complex valued interpolation is determined by :attr:`mode`.

        Currently temporal, spatial and volumetric sampling are supported, i.e.
        expected inputs are 3-D, 4-D or 5-D in shape.

        The input dimensions are interpreted in the form:
        `mini-batch x [optional channels] x [optional height] x width x 2`.

        The modes available for resizing are: `nearest`, `linear` (3D-only),
        `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

        Args:
            input (Tensor): the input tensor
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
                output spatial size.
            scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
            mode (str): algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
            align_corners (bool, optional): Geometrically, we consider the pixels of the
                input and output as squares rather than points.
                If set to ``True``, the input and output tensors are aligned by the
                center points of their corner pixels, preserving the values at the corner pixels.
                If set to ``False``, the input and output tensors are aligned by the corner
                points of their corner pixels, and the interpolation uses edge value padding
                for out-of-boundary values, making this operation *independent* of input size
                when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
                is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
                Default: ``False``
            recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
                interpolation calculation.  When `scale_factor` is passed as a parameter, it is used
                to compute the `output_size`.  If `recompute_scale_factor` is ```False`` or not specified,
                the passed-in `scale_factor` will be used in the interpolation computation.
                Otherwise, a new `scale_factor` will be computed based on the output and input sizes for
                use in the interpolation computation (i.e. the computation will be identical to if the computed
                `output_size` were passed-in explicitly).  Note that when `scale_factor` is floating-point,
                the recomputed scale_factor may differ from the one passed in due to rounding and precision
                issues.

        .. note::
            With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
            negative values or values greater than 255 for images.
            Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
            when displaying the image.

        .. warning::
            With ``align_corners = True``, the linearly interpolating modes
            (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
            output and input pixels, and thus the output values can depend on the
            input size. This was the default behavior for these modes up to version
            0.3.1. Since then, the default behavior is ``align_corners = False``.
            See :class:`~th.nn.Upsample` for concrete examples on how this
            affects the outputs.

        .. warning::
            When scale_factor is specified, if recompute_scale_factor=True,
            scale_factor is used to compute the output_size which will then
            be used to infer new scales for the interpolation.
            The default behavior for recompute_scale_factor changed to False
            in 1.6.0, and scale_factor is used in the interpolation
            calculation.

        Note:
            When using the CUDA backend, this operation may induce nondeterministic
            behaviour in its backward pass that is not easily switched off.
            Please see the notes on :doc:`/notes/randomness` for background.
        """

    dim0 = list(range(input.dim()))
    dim = dim0.copy()
    dim.insert(1, dim[-1])
    dim.pop()

    input = input.permute(dim)

    dim0[1:-1] = dim0[2:-1]
    dim0.append(1)

    return thf.interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor).permute(dim0)


if __name__ == "__main__":

    pass
