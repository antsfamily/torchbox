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

import numpy as np
import torch as th
import torchbox as tb
import torchtsa as tt
import matplotlib.pyplot as plt


P = 10
ns = 2
dim = 0
phase = 'train'
phase = 'test'
phase = input("Please input phase: ")
nepoch = 1000
device = 'cuda:1'

datafolder = '/mnt/e/DataSets/wifi/csi/'
train_file = 'wifi_channel_pc1.mat'
valid_file = 'wifi_channel_pc2.mat'
test_file = 'wifi_channel_pc1.mat'
snapshot_folder = './data/snapshot/wifichannel/'

X = th.from_numpy(tb.loadmat(datafolder + train_file)['H']).to(th.complex64)
Xtrain, Ytrain = tb.c2r(tt.lagmat(X[:-1], P, dim=0, lagdir='-->'), cdim=2, keepdim=False), tb.c2r(tt.lagmat(X[1:], P, dim=0, lagdir='-->'), cdim=2, keepdim=False)
X = th.from_numpy(tb.loadmat(datafolder + valid_file)['H']).to(th.complex64)
Xvalid, Yvalid = tb.c2r(tt.lagmat(X[:-1], P, dim=0, lagdir='-->'), cdim=2, keepdim=False), tb.c2r(tt.lagmat(X[1:], P, dim=0, lagdir='-->'), cdim=2, keepdim=False)
X = th.from_numpy(tb.loadmat(datafolder + test_file)['H']).to(th.complex64)
Xtest, Ytest = tb.c2r(tt.lagmat(X[:-1], P, dim=0, lagdir='-->'), cdim=2, keepdim=False), tb.c2r(tt.lagmat(X[1:], P, dim=0, lagdir='-->'), cdim=2, keepdim=False)

trainds = th.utils.data.TensorDataset(Xtrain, Ytrain)
validds = th.utils.data.TensorDataset(Xvalid, Yvalid)
testds = th.utils.data.TensorDataset(Xtest, Ytest)
traindl = th.utils.data.DataLoader(trainds, num_workers=4, batch_size=32, shuffle=True)
validdl = th.utils.data.DataLoader(validds, num_workers=4, batch_size=32, shuffle=False)
testdl = th.utils.data.DataLoader(testds, num_workers=4, batch_size=32, shuffle=False)

class Conv3dLSTM(th.nn.Module):
    def __init__(self, inchnl=1):
        super(Conv3dLSTM, self).__init__()

        self.convlstm1 = tb.ConvLSTM(rank=3, in_channels=inchnl, out_channels=16, kernel_size=3, batch_first=True)
        self.batchnorm1 = th.nn.BatchNorm3d(16)
        self.convlstm2 = tb.ConvLSTM(rank=3, in_channels=16, out_channels=16, kernel_size=3, batch_first=True)
        self.batchnorm2 = th.nn.BatchNorm3d(16)
        self.outconv = th.nn.Conv3d(16, 1, kernel_size=3, padding='same')
        self.outact = th.nn.ReLU()

    def forward(self, x, noising=False):

        if noising:
            x = tb.awgns(x, snrv=th.rand(1).item()*75 + 5, cdim=None, dim=(-2, -1))

        x, _ = self.convlstm1(x)
        x = self.batchnorm1(x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])).reshape(x.shape)
        x, _ = self.convlstm2(x)
        x = self.batchnorm2(x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])).reshape(x.shape)
        
        y = []
        for t in range(x.shape[1]):
            y.append(self.outconv(x[:, t, ...]))
        
        y = th.stack(y, dim=1)
        y = self.outact(y)
        return y

model = Conv3dLSTM(inchnl=2)
model = model.to(device)

losses = [tb.SSELoss(cdim=None, dim=(-2, -1))]

optimizer = th.optim.Adam(model.parameters(), lr=0.001)

scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0.00001)

th.backends.cudnn.benchmark = True
th.backends.cudnn.deterministic = True
th.backends.cuda.matmul.allow_tf32 = False
th.backends.cudnn.allow_tf32 = False

if phase == 'train':
    loss_best = float('Inf')
    losslog = tb.LossLog(snapshot_folder, filename='convlstm_losslog')
    for epoch in range(nepoch):

        lossv_train = tb.train_epoch(model, traindl, losses, optimizer, scheduler, epoch, device=device, noising=True)
        lossv_valid = tb.valid_epoch(model, validdl, losses, epoch, device=device, noising=False)
        losslog.add('train', lossv_train)
        losslog.add('valid', lossv_valid)

        if lossv_valid < loss_best:
            loss_best = lossv_valid
            tb.save_model(snapshot_folder + 'convlstm_best.pth.tar', model, epoch=epoch)
        if epoch % 10 == 0:
            losslog.plot()

if phase == 'test':
    model = tb.load_model(snapshot_folder + 'convlstm_best.pth.tar', model)
    lossv_test, Ypred = tb.test_epoch(model, testdl, losses, device=device, noising=False)
    print(tb.sse(Ypred, Ytest, dim=(-2, -1), reduction='mean'))
    N, Ts, _, _, _, _ = Ypred.shape
    Ns = min(10, N)

    fig = plt.figure()
    for t in range(Ts):
        fig.clf()
        plt = tb.imshow([Ytest[n, t, 0, :, :, 0] for n in range(Ns)] + [Ypred[n, t, 0, :, :, 0] for n in range(Ns)] + [(Ytest[n, t, 0, :, :, 0]-Ypred[n, t, 0, :, :, 0]).abs() for n in range(Ns)], nrows=3, ncols=Ns, fig=fig)
        plt.pause(0.5)
