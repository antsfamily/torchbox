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


P = 1
ns = 2
dim = 0
phase = 'train'
phase = 'test'
phase = input("Please input phase: ")
nepoch = 1000
device = 'cuda:0'

lossname = 'SSELoss'  # 'SSELoss' 'PeaCorLoss'
dataset_file = '/mnt/d/DataSets/dgi/MovingMNIST/mnist_test_seq.npy'
snapshot_folder = './data/snapshot/wifichannel/%s' % lossname

X = np.load(dataset_file)
X = th.from_numpy(X).unsqueeze(2).to(th.float32)
X = X.transpose(0, 1)
N, T, C, H, W = X.shape
Xtrain, Ytrain = tt.lagmat(X[:int(0.6*N), :-1], ns, dim=1), X[:int(0.6*N), ns:]
Xtrain, Ytrain = Xtrain.reshape(Xtrain.shape[0]*Xtrain.shape[1], ns*C, H, W), Ytrain.reshape(Ytrain.shape[0]*Ytrain.shape[1], C, H, W)
Xvalid, Yvalid = tt.lagmat(X[int(0.6*N):int(0.6*N)+int(0.1*N), :-1], ns, dim=1), X[int(0.6*N):int(0.6*N)+int(0.1*N), ns:]
Xvalid, Yvalid = Xvalid.reshape(Xvalid.shape[0]*Xvalid.shape[1], ns*C, H, W), Yvalid.reshape(Yvalid.shape[0]*Yvalid.shape[1], C, H, W)
Xtest, Ytest = tt.lagmat(X[int(0.7*N):, :-1], ns, dim=1), X[int(0.7*N):, ns:]
Xtest, Ytest = Xtest.reshape(Xtest.shape[0]*Xtest.shape[1], ns*C, H, W), Ytest.reshape(Ytest.shape[0]*Ytest.shape[1], C, H, W)

trainds = th.utils.data.TensorDataset(Xtrain, Ytrain)
validds = th.utils.data.TensorDataset(Xvalid, Yvalid)
testds = th.utils.data.TensorDataset(Xtest, Ytest)
traindl = th.utils.data.DataLoader(trainds, num_workers=4, batch_size=32, shuffle=True)
validdl = th.utils.data.DataLoader(validds, num_workers=4, batch_size=32, shuffle=False)
testdl = th.utils.data.DataLoader(testds, num_workers=4, batch_size=32, shuffle=False)

class PredCNN(th.nn.Module):
    def __init__(self, inchnl=1):
        super(PredCNN, self).__init__()

        self.conv1 = th.nn.Conv2d(in_channels=inchnl, out_channels=16, kernel_size=3, padding='same')
        self.bn1 = th.nn.BatchNorm2d(16)
        self.act1 = th.nn.ReLU()
        self.conv2 = th.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.bn2 = th.nn.BatchNorm2d(16)
        self.act2 = th.nn.ReLU()
        self.outconv = th.nn.Conv2d(16, 1, kernel_size=3, padding='same')
        self.outact = th.nn.ReLU()

    def forward(self, x, noising=False):

        if noising:
            x = tb.awgns(x, snrv=th.rand(1).item()*75 + 5, cdim=None, dim=(-2, -1))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.outconv(x)
        x = self.outact(x)

        return x

model = PredCNN(inchnl=ns)
model = model.to(device)

losses = []
lossnames = lossname.split('+')
for lname in lossnames:
    losses.append(eval('tb.'+lname)(cdim=1, dim=(-3, -2, -1), reduction='sum'))

optimizer = th.optim.Adam(model.parameters(), lr=0.001)

scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0.00001)

th.backends.cudnn.benchmark = True
th.backends.cudnn.deterministic = True
th.backends.cuda.matmul.allow_tf32 = False
th.backends.cudnn.allow_tf32 = False

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

    Ypred = Ypred.reshape(Ypred.shape[0]//(T-ns), T-ns, *Ypred.shape[1:])
    Ytest = Ytest.reshape(Ytest.shape[0]//(T-ns), T-ns, *Ytest.shape[1:])
    print(tb.sse(Ypred, Ytest, dim=(-2, -1), reduction='mean'))
    N, Ts, _, _, _ = Ypred.shape
    Ns = min(10, N)

    fig = plt.figure()
    for t in range(Ts):
        fig.clf()
        plt = tb.imshow([Ytest[n, t, 0] for n in range(Ns)] + [Ypred[n, t, 0] for n in range(Ns)] + [(Ytest[n, t, 0]-Ypred[n, t, 0]).abs() for n in range(Ns)], nrows=3, ncols=Ns, fig=fig)
        plt.pause(0.5)
