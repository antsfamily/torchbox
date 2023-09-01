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
import os
import argparse
import torch as th
import torchbox as tb
import torchtsa as tt
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='configuration')
parser.add_argument('--phase', type=str, default='test', help='train or test')
parser.add_argument('--scheme', type=str, default='scheme1', help='train and test scheme')
parser.add_argument('--device', type=str, default='', help='device')

args = parser.parse_args()

datafolder = '/mnt/e/DataSets/wifi/csi/'
train_file = 'wifi_channel_pc1.mat'
valid_file = 'wifi_channel_pc2.mat'
test_file = 'wifi_channel_pc3.mat'
# test_file = 'wifi_channel_pd3.mat'
# test_file = 'wifi_channel_n3.mat'
snapshot_root = './data/snapshot/wifichannel/predcnn/'
snapshot_folder = snapshot_root + args.scheme + '/'
logfile = snapshot_folder + '%s.log' % args.phase

sch = tb.loadyaml(snapshot_root + 'schemes.yaml', args.scheme)
device = sch['device'] if args.device == '' else args.device

X = tb.loadmat(datafolder + train_file)['H']
X = th.from_numpy(X).to(th.complex64)
# X = tb.zscore(X, dim=(1, 2, 3))
Xtrain, Ytrain = tb.c2r(tt.lagmat(X[:-1], sch['order'], dim=0, lagdir='-->'), cdim=1, keepdim=True), tb.c2r(X[sch['order']:], cdim=1, keepdim=False)
X = tb.loadmat(datafolder + valid_file)['H']
X = th.from_numpy(X).to(th.complex64)
# X = tb.zscore(X, dim=(1, 2, 3))
Xvalid, Yvalid = tb.c2r(tt.lagmat(X[:-1], sch['order'], dim=0, lagdir='-->'), cdim=1, keepdim=True), tb.c2r(X[sch['order']:], cdim=1, keepdim=False)
X = tb.loadmat(datafolder + test_file)['H']
X = th.from_numpy(X).to(th.complex64)
# X = tb.zscore(X, dim=(1, 2, 3))
Xtest, Ytest = tb.c2r(tt.lagmat(X[:-1], sch['order'], dim=0, lagdir='-->'), cdim=1, keepdim=True), tb.c2r(X[sch['order']:], cdim=1, keepdim=False)

trainds = th.utils.data.TensorDataset(Xtrain, Ytrain)
validds = th.utils.data.TensorDataset(Xvalid, Yvalid)
testds = th.utils.data.TensorDataset(Xtest, Ytest)
traindl = th.utils.data.DataLoader(trainds, num_workers=4, batch_size=512, shuffle=True)
validdl = th.utils.data.DataLoader(validds, num_workers=4, batch_size=512, shuffle=False)
testdl = th.utils.data.DataLoader(testds, num_workers=4, batch_size=64, shuffle=False)

tb.setseed(sch['seed'], target='torch')

class PredCNN(th.nn.Module):
    def __init__(self, sch):
        super(PredCNN, self).__init__()
        inchnl = 2*sch['order'] if 'order' in sch.keys() else 2
        self.outbn = sch['outbn'] if 'outbn' in sch.keys() else False
        outact = sch['outact'] if 'outact' in sch.keys() else 'Identity()'
        self.conv1 = th.nn.Conv3d(in_channels=inchnl, out_channels=16, kernel_size=3, padding='same', bias=False)
        self.bn1 = th.nn.BatchNorm3d(16)
        self.act1 = th.nn.ReLU()
        self.conv2 = th.nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding='same', bias=False)
        self.bn2 = th.nn.BatchNorm3d(16)
        self.act2 = th.nn.ReLU()
        self.outconv = th.nn.Conv3d(16, 2, kernel_size=3, padding='same', bias=False)
        if self.outbn:
            self.obn = th.nn.BatchNorm3d(2)
        self.outact = eval('th.nn.' + outact)

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
        if self.outbn:
            x = self.obn(x)
        x = self.outact(x)

        return x

model = PredCNN(sch)
model = model.to(device)

criterion = [eval(criterion) for criterion in sch['criterion']]
optimizer = eval(sch['optimizer'])
scheduler = eval(sch['scheduler'])

th.backends.cudnn.benchmark = True
th.backends.cudnn.deterministic = True
th.backends.cuda.matmul.allow_tf32 = False
th.backends.cudnn.allow_tf32 = False

# lrfinder = tb.LrFinder(device=device, plotdir='./')
# lrfinder.find(traindl, model, optimizer=optimizer, criterion=criterion, nin=1, nout=1)
# lrfinder.plot()

os.makedirs(snapshot_folder, exist_ok=True)
logf = tb.fopen(logfile, 'w')
print(sch, file=logf)
print(model, file=logf)
if args.phase == 'train':
    loss_best = float('Inf')
    model_best = tb.get_parameters(model, epoch=-1)
    losslog = tb.LossLog(snapshot_folder, filename='losslog')
    for epoch in range(sch['nepoch']):

        lossv_train = tb.train_epoch(model, traindl, criterion, optimizer, scheduler, epoch, logf=logf, device=device, noising=True)
        lossv_valid = tb.valid_epoch(model, validdl, criterion, epoch, logf=logf, device=device, noising=False)
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
    lossv_test, Ypred = tb.test_epoch(model, testdl, criterion, logf=logf, device=device, noising=False)
    Ypred = tb.r2c(Ypred, cdim=1)
    Ytest = tb.r2c(Ytest, cdim=1)
    print(tb.sse(Ypred, Ytest, cdim=None, dim=(-3, -2, -1), reduction='mean'), file=logf)
    T, _, _, _ = Ypred.shape
    logf.close()

    fig = plt.figure()
    for t in range(T):
        fig.clf()
        plt = tb.imshow([Ytest[t, :, :, 0].abs(), Ypred[t, :, :, 0].abs(), (Ytest[t, :, :, 0]-Ypred[t, :, :, 0]).abs()], nrows=1, ncols=3, fig=fig)
        plt.pause(0.5)
