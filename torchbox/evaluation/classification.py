#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : contrast.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 11 2020
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

import numpy as np
import torch as th
from torchbox.utils.colors import gray2rgb
import matplotlib.pyplot as plt


def categorical2onehot(X, nclass=None):
    r"""converts categorical to onehot

    Parameters
    ----------
    X : list or tensor
        the categorical list or tensor
    nclass : int, optional
        the number of classes, by default None (auto detected)

    Returns
    -------
    tensor
        the one-hot matrix
    """    
    X = th.tensor(X) if type(X) is not th.Tensor else X
    if th.is_floating_point(X):
        X = X.to(th.int64)
    X = X.reshape(len(X))
    if nclass is None:
        nclass = len(th.unique(X))
    
    offset = th.min(X)
    X -= offset
    return th.eye(nclass)[X]

def onehot2categorical(X, axis=-1, offset=0):
    r"""converts onehot to categorical

    Parameters
    ----------
    X : list or tensor
        the one-hot tensor
    axis : int, optional
        the axis for one-hot encoding, by default -1
    offset : int, optional
        category label head, by default 0

    Returns
    -------
    tensor
        the category label
    """    
    return th.argmax(X, axis=axis) + offset

def accuracy(P, T, axis=None):
    r"""computes the accuracy

    .. math::
       A = \frac{\sum(P==T)}{N}

    where :math:`N` is the number of samples.

    Parameters
    ----------
    P : list or tensor
        predicted label (categorical or one-hot)
    T : list or tensor
        target label (categorical or one-hot)
    axis : int, optional
        the one-hot encoding axis, by default None, which means :attr:`P` and :attr:`T` are categorical.

    Returns
    -------
    float
        the accuracy

    Raises
    ------
    ValueError
        :attr:`P` and :attr:`T` should have the same shape!
    ValueError
        You should specify the one-hot encoding axis when :attr:`P` and :attr:`T` are in one-hot formation!

    Examples
    --------

    ::

        import torchbox as tb

        T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
        P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

        print(tb.accuracy(P, T))

        #---output
        0.8 

    """
    P = th.tensor(P) if (type(P) is list) or (type(P) is tuple) else P
    T = th.tensor(T) if (type(T) is list) or (type(T) is tuple) else T

    if P.shape != T.shape:
        raise ValueError("P and T should have the same shape!")

    if ((np.prod(P.shape) != np.max(P.shape)) or (np.prod(T.shape) != np.max(T.shape))) and (axis is None):
        raise ValueError("You should specify the one-hot encoding axis!")
    
    if axis is not None:
       P = th.argmax(P, axis=axis) if np.prod(P.shape) != np.max(P.shape) else P
       T = th.argmax(T, axis=axis) if np.prod(T.shape) != np.max(T.shape) else T

    return th.sum(P==T) / len(P)


def confusion(P, T, axis=None, cmpmode='...'):
    r"""computes the confusion matrix

    Parameters
    ----------
    P : list or tensor
        predicted label (categorical or one-hot)
    T : list or tensor
        target label (categorical or one-hot)
    axis : int, optional
        the one-hot encoding axis, by default None, which means :attr:`P` and :attr:`T` are categorical.
    cmpmode : str, optional
        ``'...'`` for one-by one mode, ``'@'`` for multiplication mode (:math:`P^TT`), by default '...'

    Returns
    -------
    tensor
        the confusion matrix

    Raises
    ------
    ValueError
        :attr:`P` and :attr:`T` should have the same shape!
    ValueError
        You should specify the one-hot encoding axis when :attr:`P` and :attr:`T` are in one-hot formation!

    Examples
    --------

    ::

        import torchbox as tb

        T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
        P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

        C = tb.confusion(P, T, cmpmode='...')
        print(C)
        C = tb.confusion(P, T, cmpmode='@')
        print(C)

        #---output
        [[3. 0. 2. 0. 1. 0.]
        [0. 3. 0. 0. 0. 0.]
        [1. 0. 1. 0. 0. 0.]
        [0. 0. 0. 3. 0. 0.]
        [0. 0. 0. 0. 3. 0.]
        [0. 0. 0. 0. 0. 3.]]
        [[3. 0. 2. 0. 1. 0.]
        [0. 3. 0. 0. 0. 0.]
        [1. 0. 1. 0. 0. 0.]
        [0. 0. 0. 3. 0. 0.]
        [0. 0. 0. 0. 3. 0.]
        [0. 0. 0. 0. 0. 3.]]

    """    
    P = th.tensor(P) if (type(P) is list) or (type(P) is tuple) else P
    T = th.tensor(T) if (type(T) is list) or (type(T) is tuple) else T

    if P.shape != T.shape:
        raise ValueError("P and T should have the same shape!")

    if ((np.prod(P.shape) != np.max(P.shape)) or (np.prod(T.shape) != np.max(T.shape))) and (axis is None):
        raise ValueError("You should specify the one-hot axis!")
    
    if axis is not None:
       P = th.argmax(P, axis=axis) if np.prod(P.shape) != np.max(P.shape) else P
       T = th.argmax(T, axis=axis) if np.prod(T.shape) != np.max(T.shape) else T

    if th.is_floating_point(T):
        T = T.to(th.int64)
    if th.is_floating_point(P):
        P = P.to(th.int64)

    if cmpmode.lower() in ['...', 'for']:
        nclass = len(th.unique(T))
        offset = th.min(T)
        T -= offset
        P -= offset
        C = th.zeros((nclass, nclass))
        for p, t in zip(P, T):
            C[p, t] += 1

    if cmpmode.lower() in ['@', 'mul', 'matmul']:
        P = categorical2onehot(P, nclass=None)
        T = categorical2onehot(T, nclass=None)
        C = P.T @ T

    return C

def kappa(C):
    r"""computes kappa

    .. math::
       K = \frac{p_o - p_e}{1 - p_e}

    where :math:`p_o` and :math:`p_e` can be obtained by

    .. math::
        p_o = \frac{\sum_iC_{ii}}{\sum_i\sum_jC_{ij}} 
    
    .. math::
        p_e = \frac{\sum_j\left(\sum_iC_{ij}\sum_iC_{ji}\right)}{\sum_i\sum_jC_{ij}} 

    Parameters
    ----------
    C : Tensor
        The confusion matrix

    Returns
    -------
    float
        The kappa value.

    Examples
    --------

    ::

        import torchbox as tb

        T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
        P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

        C = tb.confusion(P, T, cmpmode='...')
        print(tb.kappa(C))
        print(tb.kappa(C.T))

        #---output
        0.7583081570996979
        0.7583081570996979

    """

    nsamples = th.sum(C)
    po = th.sum(th.diag(C)) / nsamples
    pe = th.sum(th.sum(C, axis=0) * th.sum(C.T, 0)) / nsamples / nsamples
    K = (po - pe) / (1 - pe)
    return K

def plot_confusion(C, cmap=None, mode='rich', xticks='label', yticks='label', xlabel='Target', ylabel='Predicted', title='Confusion matrix', **kwargs):
    r"""plots confusion matrix.

    plots confusion matrix.

    Parameters
    ----------
    C : Tensor
        The confusion matrix
    cmap : None or str, optional
        The colormap, by default :obj:`None`, which means our default configuration (green-coral)
    mode : str, optional
        ``'pure'``, ``'bare'``, ``'simple'`` or ``'rich'``
    xticks : str, tuple or list, optional
        ``'label'`` --> class labels, or you can specify class name list, by default ``'label'``
    yticks : str, tuple or list, optional
        ``'label'`` --> class labels, or you can specify class name list, by default ``'label'``
    xlabel : str, optional
        The label string of axis-x, by default 'Target'
    ylabel : str, optional
        The label string of axis-y, by default 'Predicted'
    title : str, optional
        The title string, by default 'Confusion matrix'
    kwargs :
        linespacing : float
            The line spacing of text, by default ``0.15``
        numftd : dict
            The font dict of integer value, by default ::

                dict(fontsize=12, color='black', 
                     family='Times New Roman', 
                     weight='bold', style='normal')
        pctftd : dict
            The font dict of percent value, by default ::
            
                dict(fontsize=12, color='black', 
                     family='Times New Roman', 
                     weight='light', style='normal')
        restftd : dict
            The font dict of label, title and ticks, by default ::

                dict(fontsize=12, color='black', 
                     family='Times New Roman', 
                     weight='light', style='normal')
        pctfmt : str
            the format of percent value, such as ``'%.xf'`` means formating with two decimal places, by default ``'%.1f'``

    Returns
    -------
    pyplot
        pyplot handle

    Example
    -------

    .. image:: ./_static/ConfusionMatrixSimple.png
       :scale: 100 %
       :align: center

    .. image:: ./_static/ConfusionMatrixRich.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torchbox as tb

        T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
        P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

        C = tb.confusion(P, T, cmpmode='@')

        plt = tb.plot_confusion(C, cmap=None, mode='simple')
        plt = tb.plot_confusion(C, cmap='summer', mode='simple')
        plt.show()

        plt = tb.plot_confusion(C, cmap=None, mode='rich')
        plt = tb.plot_confusion(C, cmap='summer', mode='rich')
        plt.show()

    """    

    nclass, _ = C.shape
    n = th.sum(C)
    linespacing = 0.15
    pctfmt = '%.1f'

    if 'linespacing' in kwargs:
        linespacing = kwargs['linespacing']
        del(kwargs['linespacing'])
    if 'pctfmt' in kwargs:
        pctfmt = kwargs['pctfmt']
        del(kwargs['pctfmt'])

    if 'numftd' in kwargs:
        numftd = kwargs['numftd']
    else:
        numftd = dict(fontsize=12,
                        color='black',
                        family='Times New Roman',
                        weight='bold',
                        style='normal',
                        )
    if 'pctftd' in kwargs:
        pctftd = kwargs['pctftd']
    else:
        pctftd = dict(fontsize=12,
                        color='black',
                        family='Times New Roman',
                        weight='light',
                        style='normal',
                        )
    if 'restftd' in kwargs:
        restftd = kwargs['restftd']
    else:
        restftd = dict(fontsize=12,
                        color='black',
                        family='Times New Roman',
                        weight='light',
                        style='normal',
                        )

    xticks = [str(i) for i in range(1, nclass+1)] if xticks == 'label' else list(xticks)
    yticks = [str(i) for i in range(1, nclass+1)] if yticks == 'label' else list(yticks)

    if mode == 'rich':
        xticks = [' '] + xticks + [' ']
        yticks = [' '] + yticks + [' ']
        plt.figure(**kwargs)
        if cmap is not None:
            Cc = gray2rgb(C, cmap=cmap)
            Cc = th.nn.functional.pad(Cc, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
            Cc = Cc.to(th.uint8)
        else:    
            Cc = th.zeros((nclass+2, nclass+2, 3), dtype=th.uint8)
            Cc[..., 0] = 249; Cc[..., 1] = 196; Cc[..., 2] = 192
            for i in range(1, nclass+1):
                Cc[i, i, :] = th.tensor([186, 231, 198])
        Cc[-1, :, :] = Cc[:, -1, :] = Cc[0, :, :] = Cc[:, 0, :] = th.tensor([240, 240, 240])
        Cc[-1, -1, :] = Cc[0, 0, :] = th.tensor([214, 217, 226])
        plt.imshow(Cc)
        
        for i in range(0, nclass+2):
            plt.plot((-0.5, nclass+1+0.5), (i-0.5, i-0.5), '-k', linewidth=0.5)
            plt.plot((i-0.5, i-0.5), (-0.5, nclass+1+0.5), '-k', linewidth=0.5)

        for i in range(nclass):
            for j in range(nclass):
                s1 = '%d' % C[i, j]
                s2 = pctfmt % (C[i, j] * 100 / n) + '%'
                plt.text(j+1, i+1-linespacing, s1, fontdict=numftd, ha='center', va='center')
                plt.text(j+1, i+1+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        numftd['color'], numftd['weight'] = 'black', 'normal'
        pctftd['color'], pctftd['weight'] = 'coral', 'normal'
        qcol = th.sum(C, axis=0)
        qrow = th.sum(C, axis=1)
        qdiag = th.diag(C)
        j = 0
        for i in range(nclass):
            s1 = '%d' % qrow[i]
            s2 = '%d' % (qrow[i] - qdiag[i])
            plt.text(j, i+1-linespacing, s1, fontdict=numftd, ha='center', va='center')
            plt.text(j, i+1+linespacing, s2, fontdict=pctftd, ha='center', va='center')
        
        i = 0
        for j in range(nclass):
            s1 = '%d' % qcol[j]
            s2 = '%d' % (qcol[j] - qdiag[j])
            plt.text(j+1, i-linespacing, s1, fontdict=numftd, ha='center', va='center')
            plt.text(j+1, i+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        s1 = '%d' % n
        s2 = '%d' % (n - sum(qdiag))
        numftd['weight'] = 'bold'
        pctftd['weight'] = 'bold'
        plt.text(0, 0-linespacing, s1, fontdict=numftd, ha='center', va='center')
        plt.text(0, 0+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        numftd['color'], numftd['weight'] = 'green', 'normal'
        pctftd['color'], pctftd['weight'] = 'coral', 'normal'
        pcol = qdiag / qcol
        prow = qdiag / qrow
        j = nclass
        for i in range(nclass):
            s1 = pctfmt % (prow[i] * 100) + '%'
            s2 = pctfmt % (100 - prow[i] * 100) + '%'
            plt.text(j+1, i+1-linespacing, s1, fontdict=numftd, ha='center', va='center')
            plt.text(j+1, i+1+linespacing, s2, fontdict=pctftd, ha='center', va='center')
        
        i = nclass
        for j in range(nclass):
            s1 = pctfmt % (pcol[j] * 100) + '%'
            s2 = pctfmt % (100 -  pcol[j] * 100) + '%'
            plt.text(j+1, i+1-linespacing, s1, fontdict=numftd, ha='center', va='center')
            plt.text(j+1, i+1+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        nd = th.trace(C) / n
        s1 = pctfmt % (nd * 100) + '%'
        s2 = pctfmt % (100 -  nd * 100) + '%'
        numftd['weight'] = 'bold'
        pctftd['weight'] = 'bold'
        plt.text(nclass+1, nclass+1-linespacing, s1, fontdict=numftd, ha='center', va='center')
        plt.text(nclass+1, nclass+1+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        plt.xticks(range(0, nclass+2), xticks, fontproperties=restftd['family'], size=restftd['fontsize'])
        plt.yticks(range(0, nclass+2), yticks, ha='center', va='center', rotation=90, fontproperties=restftd['family'], size=restftd['fontsize'])
        plt.tick_params(left=False, bottom=False)
        plt.xlabel(xlabel, fontdict=restftd)
        plt.ylabel(ylabel, fontdict=restftd)
        plt.title(title, fontdict=restftd)

    if mode == 'simple':
        xticks = xticks + [' ']
        yticks = yticks + [' ']
        plt.figure(**kwargs)
        if cmap is not None:
            Cc = gray2rgb(C, cmap=cmap)
            Cc = th.nn.functional.pad(Cc, (0, 0, 0, 1, 0, 1), mode='constant', value=0)
            Cc = Cc.to(th.uint8)
        else:    
            Cc = th.zeros((nclass+1, nclass+1, 3), dtype=th.uint8)
            Cc[..., 0] = 249; Cc[..., 1] = 196; Cc[..., 2] = 192
            for i in range(nclass):
                Cc[i, i, :] = th.tensor([186, 231, 198])
        Cc[-1, :, :] = Cc[:, -1, :] = th.tensor([240, 240, 240])
        Cc[-1, -1, :] = th.tensor([214, 217, 226])
        plt.imshow(Cc)
        
        for i in range(nclass+1):
            plt.plot((-0.5, nclass+0.5), (i-0.5, i-0.5), '-k', linewidth=0.5)
            plt.plot((i-0.5, i-0.5), (-0.5, nclass+0.5), '-k', linewidth=0.5)

        for i in range(nclass):
            for j in range(nclass):
                s1 = '%d' % C[i, j]
                s2 = pctfmt % (C[i, j] * 100 / n) + '%'
                plt.text(j, i-linespacing, s1, fontdict=numftd, ha='center', va='center')
                plt.text(j, i+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        numftd['color'] = 'green'
        numftd['weight'] = 'normal'
        pctftd['color'] = 'coral'
        pctftd['weight'] = 'normal'
        pcol = th.diag(C) / th.sum(C, axis=0)
        prow = th.diag(C) / th.sum(C, axis=1)
        j = nclass
        for i in range(nclass):
            s1 = pctfmt % (prow[i] * 100) + '%'
            s2 = pctfmt % (100 - prow[i] * 100) + '%'
            plt.text(j, i-linespacing, s1, fontdict=numftd, ha='center', va='center')
            plt.text(j, i+linespacing, s2, fontdict=pctftd, ha='center', va='center')
        
        i = nclass
        for j in range(nclass):
            s1 = pctfmt % (pcol[j] * 100) + '%'
            s2 = pctfmt % (100 -  pcol[j] * 100) + '%'
            plt.text(j, i-linespacing, s1, fontdict=numftd, ha='center', va='center')
            plt.text(j, i+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        nd = th.trace(C) / n
        s1 = pctfmt % (nd * 100) + '%'
        s2 = pctfmt % (100 -  nd * 100) + '%'
        numftd['weight'] = 'bold'
        pctftd['weight'] = 'bold'
        plt.text(nclass, nclass-linespacing, s1, fontdict=numftd, ha='center', va='center')
        plt.text(nclass, nclass+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        plt.xticks(range(0, nclass+1), xticks, fontproperties=restftd['family'], size=restftd['fontsize'])
        plt.yticks(range(0, nclass+1), yticks, ha='center', va='center', rotation=90, fontproperties=restftd['family'], size=restftd['fontsize'])
        plt.tick_params(left=False, bottom=False)
        plt.xlabel(xlabel, fontdict=restftd)
        plt.ylabel(ylabel, fontdict=restftd)
        plt.title(title, fontdict=restftd)

    if mode == 'bare':
        xticks = xticks
        yticks = yticks
        plt.figure(**kwargs)
        if cmap is not None:
            Cc = gray2rgb(C, cmap=cmap)
            Cc = Cc.to(th.uint8)
        else:
            Cc = th.zeros((nclass, nclass, 3), dtype=th.uint8)
            Cc[..., 0] = 249; Cc[..., 1] = 196; Cc[..., 2] = 192
            for i in range(nclass):
                Cc[i, i, :] = th.tensor([186, 231, 198])
        plt.imshow(Cc)
    
        for i in range(nclass):
            plt.plot((-0.5, nclass-0.5), (i-0.5, i-0.5), '-k', linewidth=0.5)
            plt.plot((i-0.5, i-0.5), (-0.5, nclass-0.5), '-k', linewidth=0.5)

        for i in range(nclass):
            for j in range(nclass):
                s1 = '%d' % C[i, j]
                s2 = pctfmt % (C[i, j] * 100 / n) + '%'
                plt.text(j, i-linespacing, s1, fontdict=numftd, ha='center', va='center')
                plt.text(j, i+linespacing, s2, fontdict=pctftd, ha='center', va='center')

        plt.xticks(range(0, nclass), xticks, fontproperties=restftd['family'], size=restftd['fontsize'])
        plt.yticks(range(0, nclass), yticks, ha='center', va='center', rotation=90, fontproperties=restftd['family'], size=restftd['fontsize'])
        plt.tick_params(left=False, bottom=False)
        plt.xlabel(xlabel, fontdict=restftd)
        plt.ylabel(ylabel, fontdict=restftd)
        plt.title(title, fontdict=restftd)

    if mode == 'pure':
        xticks = xticks
        yticks = yticks
        plt.figure(**kwargs)
        if cmap is not None:
            Cc = gray2rgb(C, cmap=cmap)
            Cc = Cc.to(th.uint8)
        else:
            Cc = th.zeros((nclass, nclass, 3), dtype=th.uint8)
            Cc[..., 0] = 249; Cc[..., 1] = 196; Cc[..., 2] = 192
            for i in range(nclass):
                Cc[i, i, :] = th.tensor([186, 231, 198])
        plt.imshow(Cc)
    
        for i in range(nclass):
            plt.plot((-0.5, nclass-0.5), (i-0.5, i-0.5), '-k', linewidth=0.5)
            plt.plot((i-0.5, i-0.5), (-0.5, nclass-0.5), '-k', linewidth=0.5)

        for i in range(nclass):
            for j in range(nclass):
                s1 = '%d' % C[i, j]
                plt.text(j, i, s1, fontdict=numftd, ha='center', va='center')

        plt.xticks(range(0, nclass), xticks, fontproperties=restftd['family'], size=restftd['fontsize'])
        plt.yticks(range(0, nclass), yticks, ha='center', va='center', rotation=90, fontproperties=restftd['family'], size=restftd['fontsize'])
        plt.tick_params(left=False, bottom=False)
        plt.xlabel(xlabel, fontdict=restftd)
        plt.ylabel(ylabel, fontdict=restftd)
        plt.title(title, fontdict=restftd)

    return plt


if __name__ == '__main__':

    import torchbox as tb

    T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5.0])
    P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5.0])
    classnames = ['cat', 'dog', 'car', 'cup', 'desk', 'baby']

    print(tb.accuracy(P, T))
    print(tb.categorical2onehot(T))

    C = tb.confusion(P, T, cmpmode='...')
    print(C)
    C = tb.confusion(P, T, cmpmode='@')
    print(C)
    print(tb.kappa(C))
    print(tb.kappa(C.T))

    plt = tb.plot_confusion(C, cmap=None, mode='pure')
    plt = tb.plot_confusion(C, cmap='summer', xticks=classnames, yticks=classnames, mode='pure')
    plt.show()

    plt = tb.plot_confusion(C, cmap=None, mode='bare')
    plt = tb.plot_confusion(C, cmap='summer', xticks=classnames, yticks=classnames, mode='bare')
    plt.show()

    plt = tb.plot_confusion(C, cmap=None, mode='simple')
    plt = tb.plot_confusion(C, cmap='summer', xticks=classnames, yticks=classnames, mode='simple')
    plt.show()

    plt = tb.plot_confusion(C, cmap=None, mode='rich')
    plt = tb.plot_confusion(C, cmap='summer', xticks=classnames, yticks=classnames, mode='rich')
    plt.show()
