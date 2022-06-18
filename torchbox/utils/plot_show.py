#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-25 19:44:35
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import matplotlib.pyplot as plt
from torchbox.base.mathops import fnab


def cplot(ca, lmod=None):
    N = len(ca)
    if lmod is None:
        lmod = '-b'
    r = np.real(ca)
    i = np.imag(ca)
    # x = np.hstack((np.zeros(N), r))
    # y = np.hstack((np.zeros(N), i))
    for n in range(N):
        plt.plot([0, r[n]], [0, i[n]], lmod)
    plt.xlabel('real')
    plt.ylabel('imag')


def plots(x, ydict, plotdir='./', xlabel='x', ylabel='y', title='', issave=False, isshow=True):
    if type(x) is th.Tensor:
        x = x.detach().cpu().numpy()
    legend = []
    plt.figure()
    plt.grid()
    for k, v in ydict.items():
        if type(v) is th.Tensor:
            v = v.detach().cpu().numpy()
        plt.plot(x, v)
        legend.append(k)
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if issave:
        plt.savefig(plotdir + ylabel + '_' + xlabel + '.png')
    if isshow:
        plt.show()
    plt.close()


class Plots:

    def __init__(self, plotdir='./', xlabel='x', ylabel='y', title='', figname=None, issave=False, isshow=True):

        self.plotdir = plotdir
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.issave = issave
        self.isshow = isshow
        if figname is None or figname == '':
            self.figname = self.plotdir + self.ylabel + '_' + self.xlabel + '.png'
        else:
            self.figname = figname

    def __call__(self, x, ydict, figname=None):

        if figname is None or figname == '':
            figname = self.figname

        if type(x) is th.Tensor:
            x = x.detach().cpu().numpy()
        legend = []
        plt.figure()
        plt.grid()
        for k, v in ydict.items():
            if type(v) is th.Tensor:
                v = v.detach().cpu().numpy()
            plt.plot(x, v)
            legend.append(k)
        plt.legend(legend)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        if self.issave:
            plt.savefig(figname)
        if self.isshow:
            plt.show()
        plt.close()


def imshow(Xs, nrows=None, ncols=None, xlabels=None, ylabels=None, titles=None, figsize=None, outfile=None, **kwargs):
    r"""show images

    This function create an figure and show images in :math:`a` rows and :math:`b` columns.

    Parameters
    ----------
    Xs : tensor, array, list or tuple
        list/tuple of image arrays/tensors, if the type is not list or tuple, wrap it.
    nrows : int, optional
        show in :attr:`nrows` rows, by default None (auto computed).
    ncols : int, optional
        show in :attr:`ncols` columns, by default None (auto computed).
    xlabels : str, optional
        labels of x-axis
    ylabels : str, optional
        labels of y-axis
    titles : str, optional
        titles
    figsize : tuple, optional
        figure size, by default None
    outfile : str, optional
        save image to file, by default None (do not save).
    kwargs : 
        see :func:`matplotlib.pyplot.imshow`

    Returns
    -------
    plt
        plot handle
    """

    if (type(Xs) is not list) and (type(Xs) is not tuple):
        Xs = [Xs]

    n = len(Xs)
    nrows, ncols = fnab(n)
    xlabels = [xlabels] * n if (type(xlabels) is str) or (xlabels is None) else xlabels
    ylabels = [ylabels] * n if (type(ylabels) is str) or (ylabels is None) else ylabels
    titles = [titles] * n if (type(titles) is str) or (titles is None) else titles
    plt.figure(figsize=figsize)
    for i, X, xlabel, ylabel, title in zip(range(n), Xs, xlabels, ylabels, titles):
        plt.subplot(nrows, ncols, i + 1)
        if type(X) is th.Tensor:
            X = X.cpu().numpy()
        plt.imshow(X, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

    if outfile is not None:
        plt.savefig(outfile)

    return plt


if __name__ == '__main__':

    import numpy as np
    N = 3

    r = np.random.rand(N)
    i = -np.random.rand(N)

    print(r)
    print(i)
    x = r + 1j * i

    cplot(x)
    plt.show()

    Ns = 100
    x = th.linspace(-1, 1, Ns)

    y = th.randn(Ns)
    f = th.randn(Ns)

    plot = Plots(plotdir='./', issave=True)
    plot(x, {'y': y, 'f': f})


    x = th.rand(3, 100, 100)
    plt = imshow([xi for xi in x])
    plt.show()


