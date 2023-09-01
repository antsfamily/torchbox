#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : demo_convlstm.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Thu Jul 20 2022
# @version   : 0.0
# @license   : GNU General Public License (GPL)
# @note      : 
# 
# The GNU General Public License (GPL)
# Copyright (C) 2013- Zhi Liu
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import time
import torch as th
from collections import OrderedDict


class MAML:
    def __init__(self, net, alpha=0.01):

        self.net = net
        self.alpha = alpha

    def copy_weights(self):

        self.adapted_weight = {
            key: val.clone()
            for key, val in self.net.state_dict().items()
        }

    def zero_grad(self,):

        for p in self.net.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def forward(self, x, adapted_weight=None, **kwards):
        # forward using support sets
        return self.net.forward(x, adapted_weight=adapted_weight, **kwards)

    def update_base(self, grads):

        adapted_weight = OrderedDict()
        for (key, val), grad in zip(self.net.named_parameters(), grads):
            adapted_weight[key] = val - self.alpha * grad

        return adapted_weight


class MetaSGD:
    def __init__(self, net):

        self.net = net
        for k, v in net.named_parameters():
            self.lrtask[k] = th.nn.Parameter(1e-3 * th.ones_like(v, requires_grad=True))

    def copy_weights(self):

        self.adapted_weight = {
            key: val.clone()
            for key, val in self.net.state_dict().items()
        }

    def zero_grad(self,):

        for p in self.net.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def forward(self, x, adapted_weight=None, **kwards):
        # forward using support sets
        return self.net.forward(x, adapted_weight=adapted_weight, **kwards)

    def update_base(self, grads):

        adapted_weight = OrderedDict()
        for (key, val), grad in zip(self.net.named_parameters(), grads):
            adapted_weight[key] = val - self.lrtask[key] * grad

        return adapted_weight


def mamls_train_epoch(mmodel, mdl, criterions, criterionws=None, optimizer=None, scheduler=None, nsteps_base=1, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """train one epoch using MAML, MetaSGD

    Parameters
    ----------
    mmodel : Module
        the network model
    mdl : MetaDataLoader
        the meta dataloader for training :math:`\{(x_s, y_s, x_q, y_q)\}`
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    optimizer : Optimizer or None
        optimizer for meta learner, default is :obj:`None`, 
        which means ``th.optim.Adam(model.parameters(), lr=0.001)``
    scheduler : LrScheduler or None, optional
        scheduler for meta learner, default is :obj:`None`, 
        which means using fixed learning rate
    nsteps_base : int, optional
        the number of fast adapt steps in inner loop, by default 1
    epoch : int or None, optional
        current epoch index, by default None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for training, by default ``'cuda:0'``
    kwargs :
        other forward args
    """

    logf = None if logf == 'terminal' else logf

    mmodel.net.train()
    criterionws = [1.] * len(criterions) if criterionws is None else criterionws

    nmb = len(mdl)
    lossv = 0.
    tstart = time.time()
    for b in range(nmb):
        xspt, yspt, xqry, yqry = mdl.next()
        xspt, yspt, xqry, yqry = xspt.to(device), yspt.to(device), xqry.to(device), yqry.to(device)

        adapted_weights = []
        for idtask in range(mdl.bstask):
            mmodel.copy_weights()
            adapted_weight= None
            # inner loop for base learner
            for _ in range(nsteps_base):
                zspti = mmodel.forward(xspt[idtask], adapted_weight=adapted_weight, **kwargs)
                loss = 0.
                for criterionw, criterion in zip(criterionws, criterions):
                    loss += criterionw * criterion(zspti, yspt[idtask])

                mmodel.zero_grad()
                grads = th.autograd.grad(loss, mmodel.net.parameters(), create_graph=True)
                adapted_weight = mmodel.update_base(grads)  # base learner adaption
            adapted_weights.append(adapted_weight)

        # meta loss
        loss = 0.
        for idtask in range(mdl.bstask):
            zspti = mmodel.forward(xqry[idtask], adapted_weight=adapted_weights[idtask], **kwargs)
            for criterionw, criterion in zip(criterionws, criterions):
                loss += criterionw * criterion(zspti, yqry[idtask])
        loss /= mdl.bstask

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # meta learner step
        if scheduler is not None:
            scheduler.step()

        lossv += loss.item()
    lossv /= nmb
    tend = time.time()

    if epoch is None:
        print('--->Train, loss: %.4f, time: %.2f' % (lossv, tend - tstart), file=logf)
    else:
        print('--->Train epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    return lossv


def mamls_valid_epoch(mmodel, mdl, criterions, criterionws=None, nsteps_base=1, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """valid one epoch using MAML, MetaSGD

    Parameters
    ----------
    mmodel : Module
        the network model
    mdl : MetaDataLoader
        the meta dataloader for valid :math:`\{(x_s, y_s, x_q, y_q)\}`
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    nsteps_base : int, optional
        the number of fast adapt steps in inner loop, by default 1
    epoch : int or None, optional
        current epoch index, by default None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for training, by default ``'cuda:0'``
    kwargs :
        other forward args
    """

    logf = None if logf == 'terminal' else logf

    mmodel.net.eval()
    criterionws = [1.] * len(criterions) if criterionws is None else criterionws

    nmb = len(mdl)
    lossv = 0.
    tstart = time.time()
    for b in range(nmb):
        xspt, yspt, xqry, yqry = mdl.next()
        xspt, yspt, xqry, yqry = xspt.to(device), yspt.to(device), xqry.to(device), yqry.to(device)

        adapted_weights = []
        for idtask in range(mdl.bstask):
            mmodel.copy_weights()
            adapted_weight= None
            # inner loop for base learner
            for _ in range(nsteps_base):
                zspti = mmodel.forward(xspt[idtask], adapted_weight=adapted_weight, **kwargs)
                loss = 0.
                for criterionw, criterion in zip(criterionws, criterions):
                    loss += criterionw * criterion(zspti, yspt[idtask])

                mmodel.zero_grad()
                grads = th.autograd.grad(loss, mmodel.net.parameters(), create_graph=True)
                adapted_weight = mmodel.update_base(grads)  # base learner adaption
            adapted_weights.append(adapted_weight)

        # meta loss
        loss = 0.
        for idtask in range(mdl.bstask):
            zspti = mmodel.forward(xqry[idtask], adapted_weight=adapted_weights[idtask], **kwargs)
            for criterionw, criterion in zip(criterionws, criterions):
                loss += criterionw * criterion(zspti, yqry[idtask])
        loss /= mdl.bstask

        lossv += loss.item()
    lossv /= nmb
    tend = time.time()

    if epoch is None:
        print('--->Valid, loss: %.4f, time: %.2f' % (lossv, tend - tstart), file=logf)
    else:
        print('--->Valid epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    return lossv


def mamls_test_epoch(mmodel, mdl, criterions, criterionws=None, nsteps_base=1, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """Test one epoch using MAML, MetaSGD

    Parameters
    ----------
    mmodel : Module
        the network model
    mdl : MetaDataLoader
        the meta dataloader for valid :math:`\{(x_s, y_s, x_q, y_q)\}`
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    nsteps_base : int, optional
        the number of fast adapt steps in inner loop, by default 1
    epoch : int or None, optional
        current epoch index, by default None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for training, by default ``'cuda:0'``
    kwargs :
        other forward args
    """
    
    logf = None if logf == 'terminal' else logf

    mmodel.net.eval()
    criterionws = [1.] * len(criterions) if criterionws is None else criterionws

    nmb = len(mdl)
    lossv = 0.
    tstart = time.time()
    for b in range(nmb):
        xspt, yspt, xqry, yqry = mdl.next()
        xspt, yspt, xqry, yqry = xspt.to(device), yspt.to(device), xqry.to(device), yqry.to(device)

        adapted_weights = []
        for idtask in range(mdl.bstask):
            mmodel.copy_weights()
            adapted_weight= None
            # inner loop for base learner
            for _ in range(nsteps_base):
                zspti = mmodel.forward(xspt[idtask], adapted_weight=adapted_weight, **kwargs)
                loss = 0.
                for criterionw, criterion in zip(criterionws, criterions):
                    loss += criterionw * criterion(zspti, yspt[idtask])

                mmodel.zero_grad()
                grads = th.autograd.grad(loss, mmodel.net.parameters(), create_graph=True)
                adapted_weight = mmodel.update_base(grads)  # base learner adaption
            adapted_weights.append(adapted_weight)

        # meta loss
        loss = 0.
        for idtask in range(mdl.bstask):
            zspti = mmodel.forward(xqry[idtask], adapted_weight=adapted_weights[idtask], **kwargs)
            for criterionw, criterion in zip(criterionws, criterions):
                loss += criterionw * criterion(zspti, yqry[idtask])
        loss /= mdl.bstask

        lossv += loss.item()
    lossv /= nmb
    tend = time.time()

    if epoch is None:
        print('--->Test, loss: %.4f, time: %.2f' % (lossv, tend - tstart), file=logf)
    else:
        print('--->Test epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    return lossv

