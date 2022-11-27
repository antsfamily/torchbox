#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : learning_rate.py
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

import math
import numpy as np
import matplotlib.pyplot as plt


def gammalr(x, k=2, t=2, a=1):
    return a * ((x ** (k - 1)) * np.exp(-x / t)) / ((t ** k) * np.math.gamma(k))


class LrFinder():

    def __init__(self, device='cpu', plotdir=None, logf=None):
        r"""init

        Initialize LrFinder.

        Parameters
        ----------
        device : str, optional
            device string: ``'cpu'``(default), ``'cuda:0'``, ``cuda:1`` ...
        plotdir : str, optional
            If it is not None, plot the loss-lr curve and save the figure,
            otherwise plot and show but not save. (the default is None).
        logf : str or None optional
            print log to terminal or file.
        """
        self.device = device
        self.plotdir = plotdir
        self.logf = logf
        self.lrs = []
        self.losses = []
        self.avg_losses = []
        self.smt_losses = []

    def plot(self, lrmod='log', loss='smoothed'):
        r"""plot the loss-lr curve

        Plot the loss-learning rate curve.

        Parameters
        ----------
        lrmod : str, optional
            ``'log'`` --> use log scale, i.e. log10(lr) instead lr. (default)
            ``'linear'`` --> use original lr.
        loss : str, optional
            Specify which type of loss will be ploted. (the default is 'smoothed')
        """

        if lrmod in ['log', 'LOG', 'Log']:
            lrs = [math.log10(x) for x in self.lrs]
            lrmod = 'Log10'
            lrunitstr = '/Log10'
        if lrmod in ['linear', 'LINEAR', 'Linear']:
            lrs = self.lrs
            lrmod = ''
            lrunitstr = ''

        if loss in ['smoothed', 'SMOOTHED', 'Smoothed']:
            loss = 'Smoothed'
            losses = self.smt_losses
        if loss in ['average', 'AVERAGE', 'Average']:
            loss = 'Average'
            losses = self.avg_losses
        if loss in ['original', 'ORIGINAL', 'Original']:
            loss = 'Original'
            losses = self.losses

        plt.figure()
        plt.plot(lrs, losses)
        plt.xlabel('Learning rate' + lrunitstr)
        plt.ylabel('Loss')
        plt.grid()

        losslr_str = loss + 'Loss_' + lrmod + 'LR.png'

        if self.plotdir is None:
            plt.show()
        else:
            plt.savefig(self.plotdir + '/' + losslr_str)
            plt.close()

    def find(self, dataloader, model, optimizer, criterion, nin=1, nout=1, nbgc=1, lr_init=1e-8, lr_final=1e2, beta=0.98, gamma=4.):
        r"""Find learning rate

        Find learning rate, see `How Do You Find A Good Learning Rate <https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html>`_ .

        During traing, two types losses are computed

        The average loss is:

        .. math::
           \rm{avg\_loss}_i=\beta * \rm{avg\_loss}_{i-1}+(1-\beta) * \rm{loss}_i

        The smoothed loss is:

        .. math::
            \rm{smt\_loss }_{i}=\frac{\rm{avg\_loss}_{i}}{1-\beta^{i+1}}

        If :math:`i > 1` and :math:`\rm{smt\_loss} > \gamma * \rm{best\_loss}`, stop.

        If :math:`\rm{smt\_loss} < \rm{best\_loss}` or :math:`i = 1`, let :math:`\rm{best\_loss} = \rm{smt\_loss}`.


        Parameters
        ----------
        dataloader : DataLoader
            The dataloader that contains a dataset for training.
        model : Module
            Your network module.
        optimizer : Optimizer
            The optimizer such as SGD, Adam...
        criterion : Loss
            The criterion/loss used for training model.
        nin : int, optional
            The number of inputs of the model,
            the first :attr:`nin` elements are inputs,
            the rest are targets(can be None) used for computing loss. (the default is 1)
        nou : int, optional
            The number of outputs of the model used for computing loss,
            it works only when the model has multiple outputs, i.e.
            the outputs is a tuple or list which has several tensor elements (>=1).
            the first :attr:`nout` elements are used for computing loss,
            the rest are ignored. (the default is 1)
        nbgc : int, optional
            The number of batches for grad cumulation (the default is 1, which means no cumulation)
        lr_init : int, optional
            The initial learning rate (the default is 1e-8)
        lr_final : int, optional
            The final learning rate (the default is 1e-8)
        beta : float, optional
            weight for weighted sum of loss (the default is 0.98)
        gamma : float, optional
            The exploding factor :math:`\gamma`. (the default is 4.)


        Returns
        -------
        lrs : list
            Learning rates during training.
        smt_losses : list
            Smoothed losses during training.
        avg_losses : list
            Average losses during training.
        losses : list
            Original losses during training.

        Examples
        --------

        ::

            device = 'cuda:1'
            # device = 'cpu'

            num_epochs = 30
            X = th.randn(100, 2, 3, 4)
            Y = th.randn(100, 1, 3, 4)

            trainds = TensorDataset(X, Y)
            # trainds = TensorDataset(X)

            model = th.nn.Conv2d(2, 1, 1)
            model.to(device)

            trainld = DataLoader(trainds, batch_size=10, shuffle=False)

            criterion = th.nn.MSELoss(reduction='mean')

            optimizer = th.optim.SGD(model.parameters(), lr=1e-1)

            lrfinder = LrFinder(device)
            # lrfinder = LrFinder(device, plotdir='./')

            lrfinder.find(trainld, model, optimizer, criterion, nin=1,
                          nbgc=1, lr_init=1e-8, lr_final=10., beta=0.98)

            lrfinder.plot(lrmod='Linear')
            lrfinder.plot(lrmod='Log')
        """

        num = len(dataloader) - 1
        mult = (lr_final / lr_init) ** (1. / num)
        lr = lr_init
        optimizer.param_groups[0]['lr'] = lr
        avg_loss, best_loss = 0., 0.
        criterion = [criterion] if (type(criterion) is not list) and (type(criterion) is not tuple) else criterion
        for b, data in enumerate(dataloader):
            data = [x.to(self.device) for x in data]

            if b % nbgc == 0:
                optimizer.zero_grad()

            outputs = model(*data[:nin])
            loss = 0.
            for lossf in criterion:
                if type(outputs) is tuple or type(outputs) is list:
                    loss += lossf(*outputs[:nout], *data[nin:])
                else:
                    loss += lossf(outputs, *data[nin:])

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1. - beta) * loss.item()
            smt_loss = avg_loss / (1. - beta**(b + 1))

            self.losses.append(loss.item())
            self.avg_losses.append(avg_loss)

            # Stop if the loss is exploding
            if b > 1 and smt_loss > gamma * best_loss:
                return self.lrs, self.smt_losses, self.avg_losses, self.losses

            # Record the best loss
            if smt_loss < best_loss or b == 1:
                best_loss = smt_loss

            # Store the values
            self.smt_losses.append(smt_loss)
            self.lrs.append(lr)

            # Do the SGD step
            loss.backward()
            optimizer.step()

            # Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
            print('--->batch: %d, average loss: % .4f, smooth loss: %.4f, lr: %.12f ' % (b, avg_loss, smt_loss, lr), file=self.logf)
        return self.lrs, self.smt_losses, self.avg_losses, self.losses


if __name__ == '__main__':

    import torch as th
    from torch.utils.data import DataLoader, TensorDataset

    device = 'cuda:1'
    # device = 'cpu'

    num_epochs = 30
    X = th.randn(100, 2, 3, 4)
    Y = th.randn(100, 1, 3, 4)

    trainds = TensorDataset(X, Y)
    # trainds = TensorDataset(X)

    model = th.nn.Conv2d(2, 1, 1)
    model.to(device)

    trainld = DataLoader(trainds, batch_size=10, shuffle=False)

    criterion = th.nn.MSELoss(reduction='mean')

    optimizer = th.optim.SGD(model.parameters(), lr=1e-1)

    lrfinder = LrFinder(device)
    # lrfinder = LrFinder(device, plotdir='./')

    lrfinder.find(trainld, model, optimizer, criterion, nin=1,
                  nbgc=1, lr_init=1e-8, lr_final=10., beta=0.98)

    lrfinder.plot(lrmod='Linear')
    lrfinder.plot(lrmod='Log')
