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


P = 3
dim = 0
phase = 'train'
phase = 'test'
phase = input("Please input phase: ")
nepoch = 1001
seed = 2023
device = 'cpu'

lossname = 'PeaCorLoss'  # 'SSELoss' 'PeaCorLoss'
datafolder = '/mnt/e/DataSets/wifi/csi/'
train_file = 'wifi_channel_pc1.mat'
valid_file = 'wifi_channel_pc2.mat'
test_file = 'wifi_channel_pc3.mat'
snapshot_folder = './data/snapshot/wifichannel/%s' % lossname

X = th.from_numpy(tb.loadmat(datafolder + train_file)['H']).to(th.complex64)
# X = tb.zscore(X, dim=(1, 2, 3))
Xtrain, Ytrain = tb.c2r(tt.lagmat(X[:-1], P, dim=0, lagdir='-->'), cdim=1, keepdim=True), tb.c2r(X[P:], cdim=1, keepdim=False)
X = th.from_numpy(tb.loadmat(datafolder + valid_file)['H']).to(th.complex64)
# X = tb.zscore(X, dim=(1, 2, 3))
Xvalid, Yvalid = tb.c2r(tt.lagmat(X[:-1], P, dim=0, lagdir='-->'), cdim=1, keepdim=True), tb.c2r(X[P:], cdim=1, keepdim=False)
X = th.from_numpy(tb.loadmat(datafolder + test_file)['H']).to(th.complex64)
# X = tb.zscore(X, dim=(1, 2, 3))
Xtest, Ytest = tb.c2r(tt.lagmat(X[:-1], P, dim=0, lagdir='-->'), cdim=1, keepdim=True), tb.c2r(X[P:], cdim=1, keepdim=False)

trainds = th.utils.data.TensorDataset(Xtrain, Ytrain)
validds = th.utils.data.TensorDataset(Xvalid, Yvalid)
testds = th.utils.data.TensorDataset(Xtest, Ytest)
traindl = th.utils.data.DataLoader(trainds, num_workers=4, batch_size=512, shuffle=True)
validdl = th.utils.data.DataLoader(validds, num_workers=4, batch_size=512, shuffle=False)
testdl = th.utils.data.DataLoader(testds, num_workers=4, batch_size=64, shuffle=False)

tb.setseed(seed, target='torch')

class PredCNN(th.nn.Module):
    def __init__(self, inchnl=1):
        super(PredCNN, self).__init__()

        self.conv1 = th.nn.Conv3d(in_channels=inchnl, out_channels=16, kernel_size=3, padding='same', bias=False)
        self.bn1 = th.nn.BatchNorm3d(16)
        self.act1 = th.nn.ReLU()
        self.conv2 = th.nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding='same', bias=False)
        self.bn2 = th.nn.BatchNorm3d(16)
        self.act2 = th.nn.ReLU()
        self.outconv = th.nn.Conv3d(16, 2, kernel_size=3, padding='same', bias=False)
        self.outact = th.nn.Identity()

    def forward(self, x, noising=False):

        if noising:
            x = tb.awgns(x, snrv=th.rand(1).item()*75 + 5, cdim=None, dim=(-3, -2, -1))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.outconv(x)
        x = self.outact(x)

        return x

model = PredCNN(inchnl=2*P)
model = model.to(device)

losses = []
lossnames = lossname.split('+')
for lname in lossnames:
    losses.append(eval('tb.'+lname)(cdim=1, dim=(-3, -2, -1), reduction='sum'))

optimizer = th.optim.Adam(model.parameters(), lr=0.1)

scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0.00001)

th.backends.cudnn.benchmark = True
th.backends.cudnn.deterministic = True
th.backends.cuda.matmul.allow_tf32 = False
th.backends.cudnn.allow_tf32 = False

# lrfinder = tb.LrFinder(device=device, plotdir='./')
# lrfinder.find(traindl, model, optimizer=optimizer, criterion=losses, nin=1, nout=1)
# lrfinder.plot()

if phase == 'train':
    loss_best = float('Inf')
    model_best = tb.get_parameters(model, epoch=-1)
    losslog = tb.LossLog(snapshot_folder, filename='predcnn_losslog')
    for epoch in range(nepoch):

        lossv_train = tb.train_epoch(model, traindl, losses, optimizer, scheduler, epoch, device=device, noising=True)
        lossv_valid = tb.valid_epoch(model, validdl, losses, epoch, device=device, noising=False)
        losslog.add('train', lossv_train)
        losslog.add('valid', lossv_valid)

        if lossv_valid < loss_best:
            loss_best = lossv_valid
            model_best = tb.get_parameters(model, epoch=epoch)
        if epoch % 100 == 0:
            losslog.plot()
            tb.save_model(snapshot_folder + '/predcnn_best.pth.tar', model_best, epoch=epoch)
    tb.save_model(snapshot_folder + '/predcnn_final.pth.tar', model, epoch=epoch)

if phase == 'test':
    model = tb.load_model(snapshot_folder + '/predcnn_best.pth.tar', model)
    lossv_test, Ypred = tb.test_epoch(model, testdl, losses, device=device, noising=False)
    Ypred = tb.r2c(Ypred, cdim=1)
    Ytest = tb.r2c(Ytest, cdim=1)
    print(tb.sse(Ypred, Ytest, cdim=None, dim=(-3, -2, -1), reduction='mean'))
    T, _, _, _ = Ypred.shape

    fig = plt.figure()
    for t in range(T):
        fig.clf()
        plt = tb.imshow([Ytest[t, :, :, 0].abs(), Ypred[t, :, :, 0].abs(), (Ytest[t, :, :, 0]-Ypred[t, :, :, 0]).abs()], nrows=1, ncols=3, fig=fig)
        plt.pause(0.5)
