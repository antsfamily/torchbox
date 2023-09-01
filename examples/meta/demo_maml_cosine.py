#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : mnist.py
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

import os
import argparse
import torch as th
import torchbox as tb


ntask = 100
bstask = 4
kshot = 1
kquery = 1
nsteps_base = 2
Ns = 50

parser = argparse.ArgumentParser(description='configuration')
parser.add_argument('--phase', type=str, default='test', help='train or test')
parser.add_argument('--scheme', type=str, default='scheme1', help='train and test scheme')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--seed', type=int, default=2023, help='random seed')

args = parser.parse_args()

snapshot_root = 'data/snapshot/cosine/maml/'
snapshot_folder = snapshot_root + args.scheme + '/'
logfile = snapshot_folder + '%s.log' % args.phase

sch = tb.loadyaml(snapshot_root + 'schemes.yaml', args.scheme)
device = sch['device'] if args.device == '' else args.device
seed = sch['seed'] if args.seed == '' else args.seed
tb.setseed(seed=sch['seed'], target='torch')

costask = tb.CosineTask(ntask=ntask, arange=[0.1, 5.], frange=[1./(2*th.pi), 1./(2*th.pi)], prange=[0., 2*th.pi], trange=[-0.5, 0.5, Ns])
ytrain1, xtrain1 = costask.mktask(ntask=ntask, device='cpu', rett=True)
ytrain2, xtrain2 = costask.mktask(ntask=ntask, trange=[-0.5, 0.5, Ns], device='cpu', rett=True)
ytrain, xtrain = th.stack((ytrain1, ytrain2), dim=1), th.stack((xtrain1, xtrain2), dim=1)
ytrain, xtrain = ytrain.unsqueeze(1).unsqueeze(-1), xtrain.unsqueeze(1).unsqueeze(-1)

costask = tb.CosineTask(ntask=ntask, arange=[0.1, 5.], frange=[1./(2*th.pi), 1./(2*th.pi)], prange=[0., 2*th.pi], trange=[-0.5, 0.5, Ns])
yvalid1, xvalid1 = costask.mktask(ntask=ntask, device='cpu', rett=True)
yvalid2, xvalid2 = costask.mktask(ntask=ntask, trange=[-0.5, 0.5, Ns], device='cpu', rett=True)
yvalid, xvalid = th.stack((yvalid1, yvalid2), dim=1), th.stack((xvalid1, xvalid2), dim=1)
yvalid, xvalid = yvalid.unsqueeze(1).unsqueeze(-1), xvalid.unsqueeze(1).unsqueeze(-1)

costask = tb.CosineTask(ntask=ntask, arange=[0.1, 5.], frange=[1./(2*th.pi), 1./(2*th.pi)], prange=[0., 2*th.pi], trange=[-0.5, 0.5, Ns], tmode='sequential')
ytest1, xtest1 = costask.mktask(ntask=ntask, device='cpu', rett=True)
ytest2, xtest2 = costask.mktask(ntask=ntask, trange=[-0.5, 0.5, Ns], device='cpu', rett=True)
ytest, xtest = th.stack((ytest1, ytest2), dim=1), th.stack((xtest1, xtest2), dim=1)
ytest, xtest = ytest.unsqueeze(1).unsqueeze(-1), xtest.unsqueeze(1).unsqueeze(-1)

class SineNet(th.nn.Module):
    def __init__(self, nin=1) -> None:
        super(SineNet, self).__init__()
        self.nin = nin
        self.layers = th.nn.Sequential(
            th.nn.Linear(nin, 40),
            th.nn.ReLU(),
            th.nn.Linear(40, 40),
            th.nn.ReLU(),
            th.nn.Linear(40, 1),
        )

    def forward(self, x, adapted_weight=None, **kwards):
        if adapted_weight is None:
            return self.layers(x)
        else:
            x = th.nn.functional.linear(x, adapted_weight['layers.0.weight'], adapted_weight['layers.0.bias'])
            x = th.nn.functional.relu(x)
            x = th.nn.functional.linear(x, adapted_weight['layers.2.weight'], adapted_weight['layers.2.bias'])
            x = th.nn.functional.relu(x)
            x = th.nn.functional.linear(x, adapted_weight['layers.4.weight'], adapted_weight['layers.4.bias'])
            return x


mdltrain = tb.MetaDataLoader(xtrain, ytrain, bstask=bstask, nway=1, kshot=kshot, kquery=kquery, sfltask=True, sflpoint=True, dsname='train')
mdlvalid = tb.MetaDataLoader(xvalid, yvalid, bstask=bstask, nway=1, kshot=kshot, kquery=kquery, sfltask=False, sflpoint=False, dsname='valid')
mdltest = tb.MetaDataLoader(xtest, ytest, bstask=bstask, nway=1, kshot=kshot, kquery=kquery, sfltask=False, sflpoint=False, dsname='test')

model = SineNet(nin=1).to(device)
mmodel = tb.MAML(model, alpha=sch['lr_base'])
mmodel.copy_weights()

criterion = [eval(criterion) for criterion in sch['criterion']]
optimizer = eval(sch['optimizer'])
scheduler = eval(sch['scheduler'])

th.backends.cudnn.benchmark = True
th.backends.cudnn.deterministic = True
th.backends.cuda.matmul.allow_tf32 = False
th.backends.cudnn.allow_tf32 = False

os.makedirs(snapshot_folder, exist_ok=True)
logf = tb.fopen(logfile, 'w')
print(sch, file=logf)
print(model, file=logf)
if args.phase == 'train':
    loss_best = float('Inf')
    model_best = tb.get_parameters(model, epoch=-1)
    losslog = tb.LossLog(snapshot_folder, filename='losslog')
    for epoch in range(sch['nepoch']):
        
        lossv_train = tb.mamls_train_epoch(mmodel, mdltrain, criterions=criterion, optimizer=optimizer, scheduler=scheduler, nsteps_base=nsteps_base, epoch=epoch, device=device)
        lossv_valid = tb.mamls_valid_epoch(mmodel, mdlvalid, criterions=criterion, optimizer=optimizer, scheduler=scheduler, nsteps_base=nsteps_base, epoch=epoch, device=device)
        losslog.add('train', lossv_train)
        losslog.add('valid', lossv_valid)

        if lossv_valid < loss_best:
            loss_best = lossv_valid
            model_best = tb.get_parameters(model, epoch=epoch)
        if epoch % 100 == 0:
            losslog.plot()
            tb.save_model(snapshot_folder + 'best.pth.tar', model_best, epoch=epoch)
        tb.save_model(snapshot_folder + 'final.pth.tar', model, epoch=epoch)
        logf.close()

if args.phase == 'test':
    model = tb.load_model(snapshot_folder + 'best.pth.tar', model)
    lossv_test = tb.mamls_test_epoch(mmodel, mdltest, criterions=criterion, optimizer=optimizer, scheduler=scheduler, nsteps_base=2, epoch=None, device=device)
