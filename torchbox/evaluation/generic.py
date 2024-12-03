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
import torchbox as tb
import matplotlib.pyplot as plt


def geval(P, G, tol):
    r"""generic evaluation function

    Parameters
    ----------
    P : list
        The predicted results, e.g. [(attribute1, attribute2, ...), (attribute1, attribute2, ...), ...]
    G : list
        The groundtruth results, e.g. [(attribute1, attribute2, ...), (attribute1, attribute2, ...), ...]
    tol : list
        The error tolerance for each attribute
    """    
    
    lenG, lenP = len(G), len(P)

    flagG = np.array([False] * lenG)
    flagP = np.array([True] * lenP)
    errs = []
    for i, g in enumerate(G):
        for j, p in enumerate(P):
            if flagP[j]:
                matchedflagij, err = True, []
                for gk, pk, tolk in zip(g, p, tol):  # attribute
                    if tolk == 0:
                        if pk != gk:
                            matchedflagij = False
                        else:
                            err.append(0)
                    else:
                        errk = abs(gk - pk)
                        if errk > tolk:
                            matchedflagij = False
                            break
                        else:
                            err.append(errk)
                if matchedflagij:
                    flagG[i] = True
                    flagP[j] = False
                    errs.append(err)
                    break

    TP = sum(flagG == True)
    FP = sum(flagP == True)
    FN = sum(flagG == False)
    Recall = TP / (TP + FN + tb.EPS)
    Precision = TP / (TP + FP + tb.EPS)
    FAR = FP / (TP + FP + tb.EPS)
    MAR = FN / (TP + FN + tb.EPS)
    MSE = np.mean(np.array(errs)**2)

    return Recall, Precision, FAR, MAR, MSE


def eprint(rslt, fmt='%.4f'):
    r"""print evaluation result

    Parameters
    ----------
    rslt : dict
        evaluation result dict
    fmt : str, optional
        print formation of metric value, by default ``'%.4f'``

    """
    if type(rslt) is dict:
        methods = list(rslt.keys())
        maxlenmethod = 0
        for method in methods:
            if len(method) > maxlenmethod:
                maxlenmethod = len(method)
        for method, metricvalue in rslt.items():
            printstr = '%' + str(maxlenmethod) + 's--->'
            printstr = printstr % method
            for metric, value in metricvalue.items():
                meanv = np.mean(value)
                printstr += ('%s: ' + fmt + ', ') % (metric, meanv)
            print(printstr)
    else:
        print(rslt)

    return 0


def eplot(rslt, mode='vbar', xlabel=None, ylabel=None, title='Average performance of %d experiments', **kwargs):
    r"""plots evaluation results.

    plots evaluation results. If the results contain many experiments, it will be averaged.

    Parameters
    ----------
    rslt : dict
        The result dict of evaluation, {'Method1': {'Metric1': [...], 'Metric2': [...], ...}, 'Method2': {'Metric1': [...], 'Metric2': [...], ...}}
    mode : str, optional
        ``'mat'``, ``'barh'`` or ``'hbar'``, ``'barv'`` or ``'vbar'`` (default).
    xlabel : str, optional
        The label string of axis-x, by default :obj:`None` (if :attr:`mode` is ``'mat'``, ``'barv'`` or ``'vbar'``, :attr:`xlabel` is empty;
        if :attr:`mode` is ``'barh'`` or ``'hbar'``, :attr:`xlabel` is ``'Score'``.)
    ylabel : str, optional
        The label string of axis-y, by default :obj:`None` (if :attr:`mode` is ``'mat'``, ``'barh'`` or ``'hbar'``, :attr:`ylabel` is empty;
        if :attr:`mode` is ``'barv'`` or ``'vbar'``, :attr:`ylabel` is ``'Score'``.)
    title : str, optional
        The title string, by default ``'Average performance of %d experiments'``
    kwargs :
        cmap: str or None
            The colormap, by default :obj:`None`, which means our default configuration (green-coral)
            see :func:`~torchbox.utils.colors.rgb2gray` for available colormap str.
        colors: list or None
            the color for different method, only work when mode is bar, by default `None`
        grid: bool
            plot grid?, by default :obj:`False`
        bwidth: float
            The width of bar, by default ``0.5``
        bheight: float
            The height of bar, by default ``0.5``
        bspacing: float
            The spacing between bars, by default ``0.1``
        strftd : dict
            The font dict of label, title, method or metric names, by default ::

                dict(fontsize=12, color='black', 
                     family='Times New Roman', 
                     weight='light', style='normal')
        mvftd : dict
            The font dict of metric value, by default ::
            
                dict(fontsize=12, color='black', 
                     family='Times New Roman', 
                     weight='light', style='normal')
        mvfmt : str or None
            the format of metric value, such as ``'%.xf'`` means formating with two decimal places, by default ``'%.2f'``
            If :obj:`None`, no label.
        mvnorm: bool
            normalize the maximum metric value to 1? by default :obj:`False`

    Returns
    -------
    pyplot
        pyplot handle

        
    Examples
    --------

    ::

        import torchbox as tb

        result = {'Method1': {'Metric1': [1, 1.1, 1.2], 'Metric2': [2.1, 2.2, 2.3]}, 'Method2': {'Metric1': [11, 11.1, 11.2], 'Metric2': [21.1, 21.2, 21.3]}}
        tb.eprint(result)

        plt = tb.eplot(result, mode='mat')
        plt.show()
        plt = tb.eplot(result, mode='mat', mvnorm=True)
        plt.show()

        plt = tb.eplot(result, mode='mat', cmap='summer')
        plt.show()
        plt = tb.eplot(result, mode='mat', cmap='summer', mvnorm=True)
        plt.show()

        plt = tb.eplot(result, mode='vbar', bheight=0.5)
        plt.show()
        plt = tb.eplot(result, mode='vbar', bheight=0.5, mvnorm=True)
        plt.show()

        plt = tb.eplot(result, mode='hbar', bwidth=0.5)
        plt.show()
        plt = tb.eplot(result, mode='hbar', bwidth=0.5, mvnorm=True)
        plt.show()


    """    

    if 'grid' in kwargs:
        grid = kwargs['grid']
    else:
        grid = False

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = None

    if 'colors' in kwargs:
        colors = kwargs['colors']
    else:
        colors = None

    if 'bwidth' in kwargs:
        bwidth = kwargs['bwidth']
    else:
        bwidth = 0.25

    if 'bheight' in kwargs:
        bheight = kwargs['bheight']
    else:
        bheight = 0.25

    if 'bspacing' in kwargs:
        bspacing = kwargs['bspacing']
    else:
        bspacing = 0.1

    if 'strftd' in kwargs:
        strftd = kwargs['strftd']
    else:
        strftd = dict(fontsize=12,
                        color='black',
                        family='Times New Roman',
                        weight='light',
                        style='normal',
                        )
        
    if 'mvftd' in kwargs:
        mvftd = kwargs['mvftd']
    else:
        mvftd = dict(fontsize=12,
                        color='black',
                        family='Times New Roman',
                        weight='light',
                        style='normal',
                        )
        
    if 'mvfmt' in kwargs:
        mvfmt = kwargs['mvfmt']
    else:
        mvfmt = '%.2f'

    if 'mvnorm' in kwargs:
        mvnorm = kwargs['mvnorm']
    else:
        mvnorm = False

    methods = list(rslt.keys())
    metrics = list(rslt[methods[0]].keys())
    nmethod = len(methods)
    nmetric = len(metrics)

    nexps = len(rslt[methods[0]][metrics[0]])
    plotdict, plotarray = {}, np.zeros((nmethod, nmetric))
    i = 0
    for method, metricvalue in rslt.items():
        plotdict[method] = []
        j = 0
        for metric, value in metricvalue.items():
            if type(value) is th.Tensor:
                value = value.cpu().numpy()
            meanv = np.mean(value)
            plotdict[method].append(meanv)
            plotarray[i, j] = meanv
            j += 1
        i += 1

    maxmetricvalue = np.max(plotarray, axis=0, keepdims=True)
    if mvnorm:
        plotarray = plotarray / maxmetricvalue
    maxmetricvalue = maxmetricvalue.squeeze(0)

    if mode.lower() in ['barh', 'hbar']:
        xlabel = 'Score' if xlabel is None else xlabel
        ylabel = '' if ylabel is None else ylabel
        idx = 0
        fig, ax = plt.subplots(layout='constrained')
        y = np.linspace(0, (bheight*nmethod)*nmetric+bspacing*(nmetric-1), nmetric, endpoint=False)  # the bar locations
        for method, metricvalues in plotdict.items():
            offset = bheight *idx
            color = None if colors is None else colors[idx]
            if mvnorm:
                metricvalues /= maxmetricvalue
            if mvfmt is not None:
                rects = ax.barh(y + offset, height=bheight, width=metricvalues, label=method, color=color)
            ax.bar_label(rects, fmt=mvfmt, rotation=90, padding=3)
            idx += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel(xlabel, fontdict=strftd)
        ax.set_ylabel(ylabel, fontdict=strftd)
        ax.set_title(title % nexps, fontdict=strftd)
        ax.set_yticks(y+(nmethod-1)*bheight/2, metrics, ha='center', va='center', rotation=90, fontproperties=strftd['family'], size=strftd['fontsize'])
        ax.legend()
        if grid:
            ax.grid(True)

    if mode.lower() in ['barv', 'vbar']:
        xlabel = '' if xlabel is None else xlabel
        ylabel = 'Score' if ylabel is None else ylabel
        idx = 0
        fig, ax = plt.subplots(layout='constrained')
        x = np.linspace(0, (bwidth*nmethod)*nmetric+bspacing*(nmetric-1), nmetric, endpoint=False)  # the bar locations
        for method, metricvalues in plotdict.items():
            offset = bwidth *idx
            color = None if colors is None else colors[idx]
            if mvnorm:
                metricvalues /= maxmetricvalue
            if mvfmt is not None:
                rects = ax.bar(x + offset, height=metricvalues, width=bwidth, label=method, color=color)
            ax.bar_label(rects, fmt=mvfmt, padding=3)
            idx += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel(xlabel, fontdict=strftd)
        ax.set_ylabel(ylabel, fontdict=strftd)
        ax.set_title(title % nexps, fontdict=strftd)
        ax.set_xticks(x+(nmethod-1)*bwidth/2, metrics, fontproperties=strftd['family'], size=strftd['fontsize'])
        ax.legend()
        if grid:
            ax.grid(True)


    if mode.lower() in ['mat', 'matrix']:
        xticks = metrics
        yticks = methods
        C = th.from_numpy(plotarray)
        plt.figure()
        if cmap is not None:
            Cc = tb.gray2rgb(C*255, cmap=cmap, drange=(0, 255))
            Cc = Cc.to(th.uint8)
        else:
            Cc = th.zeros((nmethod, nmetric, 3), dtype=th.uint8)
            Cc[..., 0] = 249; Cc[..., 1] = 196; Cc[..., 2] = 192
        plt.imshow(Cc)
    
        for i in range(nmethod):
            plt.plot((-0.5, nmethod-0.5), (i-0.5, i-0.5), '-k', linewidth=0.5)
        for i in range(nmetric):
            plt.plot((i-0.5, i-0.5), (-0.5, nmetric-0.5), '-k', linewidth=0.5)

        for i in range(nmethod):
            for j in range(nmetric):
                s1 = mvfmt % C[i, j]
                plt.text(j, i, s1, fontdict=mvftd, ha='center', va='center')

        plt.xticks(range(0, nmethod), xticks, fontproperties=strftd['family'], size=strftd['fontsize'])
        plt.yticks(range(0, nmetric), yticks, ha='center', va='center', rotation=90, fontproperties=strftd['family'], size=strftd['fontsize'])
        plt.tick_params(left=False, bottom=False)
        plt.title(title % nexps, fontdict=strftd)



    return plt


if __name__ == "__main__":


    P = [(1, 2.0), (3, 4.0), (5, 6.9)]
    G = [(1, 2.1), (3, 4.3)]

    print(P)
    print(G)
    print(geval(P, G, tol=(0.5, 0.5)))

    P = [('cat', 1, 2.0), ('dog', 3, 4.0), ('bird', 5, 6.9)]
    G = [('cat', 1, 2.0), ('cat', 3, 4.3)]

    print(P)
    print(G)
    print(geval(P, G, tol=(0, 0.5, 0.5)))

    result = {'Method1': {'Metric1': [1, 1.1, 1.2], 'Metric2': [2.1, 2.2, 2.3]}, 'Method2': {'Metric1': [11, 11.1, 11.2], 'Metric2': [21.1, 21.2, 21.3]}}
    eprint(result)

    plt = eplot(result, mode='mat')
    plt.show()
    plt = eplot(result, mode='mat', mvnorm=True)
    plt.show()

    plt = eplot(result, mode='mat', cmap='summer')
    plt.show()
    plt = eplot(result, mode='mat', cmap='summer', mvnorm=True)
    plt.show()

    plt = eplot(result, mode='vbar', bheight=0.5, grid=True)
    plt.show()
    plt = eplot(result, mode='vbar', bheight=0.5, mvnorm=True)
    plt.show()

    plt = eplot(result, mode='hbar', bwidth=0.5, colors=['b', 'r'])
    plt.show()
    plt = eplot(result, mode='hbar', bwidth=0.5, mvnorm=True)
    plt.show()
