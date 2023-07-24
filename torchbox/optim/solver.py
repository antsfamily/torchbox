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
import torchbox as tb


def train_epoch(model, dl, losses, optimizer, scheduler, epoch, logf='terminal', device='cuda:0', **kwargs):
    r"""train one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the training dataloader
    losses : list
        a list torch.nn.Loss instances
    optimizer : function handle
        an instance of torch.optim.Optimizer
    scheduler : function handle
        an instance of torch.optim.LrScheduler
    epoch : int
        epoch index
    logf : str, optional
        IO for print log, file path or 'terminal', by default 'terminal'
    device : str, optional
        device for training, by default 'cuda:0'
    kwargs :
        other forward args
    """

    model.train()
    logf = None if logf == 'terminal' else logf

    tstart = time.time()
    lossv = 0.
    for batch_idx, (data, target) in enumerate(dl):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model.forward(data, **kwargs)

        loss = 0.
        for lossi in losses:
            loss += lossi(output, target)
        
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        lossv += loss.item()
    lossv /= len(dl.dataset)
    tend = time.time()
    print('--->Train epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    return lossv

def valid_epoch(model, dl, losses, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    r"""valid one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the validation dataloader
    losses : list
        a list torch.nn.Loss instances
    epoch : int
        epoch index,  default is None
    logf : str, optional
        IO for print log, file path or 'terminal', by default 'terminal'
    device : str, optional
        device for validation, by default 'cuda:0'
    kwargs :
        other forward args
    """

    model.eval()
    logf = None if logf == 'terminal' else logf

    tstart = time.time()
    lossv = 0.
    with th.no_grad():
        for batch_idx, (data, target) in enumerate(dl):
            data, target = data.to(device), target.to(device)

            output = model.forward(data, **kwargs)

            loss = 0.
            for lossi in losses:
                loss += lossi(output, target)
            
            lossv += loss.item()
    lossv /= len(dl.dataset)
    tend = time.time()
    if epoch is None:
        print('--->Valid, loss: %.4f, time: %.2f' % (lossv, tend - tstart), file=logf)
    else:
        print('--->Valid epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    return lossv

def test_epoch(model, dl, losses, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """Test one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the testing dataloader
    losses : list
        a list torch.nn.Loss instances
    epoch : int or None
        epoch index,  default is None
    logf : str, optional
        IO for print log, file path or 'terminal', by default 'terminal'
    device : str, optional
        device for testing, by default 'cuda:0'
    kwargs :
        other forward args
    """

    model.eval()
    logf = None if logf == 'terminal' else logf

    tstart = time.time()
    lossv = 0.
    outputs = []
    with th.no_grad():
        for batch_idx, (data, target) in enumerate(dl):
            data, target = data.to(device), target.to(device)

            output = model.forward(data, **kwargs)
            outputs.append(output.detach().cpu())

            loss = 0.
            for lossi in losses:
                loss += lossi(output, target)
            
            lossv += loss.item()
    lossv /= len(dl.dataset)
    tend = time.time()
    if epoch is None:
        print('--->Test, loss: %.4f, time: %.2f' % (lossv, tend - tstart), file=logf)
    else:
        print('--->Test epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    outputs = th.cat(outputs, dim=0)
    return lossv, outputs
