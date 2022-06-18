#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 10:38:13
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torchsar as ts
import torchbox as tb
import torch as th
import matplotlib.pyplot as plt

ftshift = True
seed = 2020

datafile = '../../../data/sarimg/ALPSRP020160970_Vr7180_R3_FocusedEach_5x256x256_AutoFocusPolyPhiMinEntropy_Epoch1000.h5'
data = tb.loadh5(datafile)
SI, ca, cr = data['SI'], data['ca'], data['cr']
N, Na, Nr, Nc = SI.shape
print(SI.shape)
x = th.from_numpy(SI[0]).unsqueeze(0)
# x = x / x.abs().max()
print(x.shape, "++++++")

carange = [[200, 2], [400, 10]]
carange = [[1, 2], [2, 4]]

crrange = [[-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]]

ppeg = ts.PolyPhaseErrorGenerator(carange, crrange, seed)

# ca, cr = ppeg.mkpec(n=1, seed=None)

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode='fftfreq')

xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode='fftfreq')

pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)

print(pa.shape, pr.shape, x.shape)
f = ts.focus(x, pa[[0]], pr[[0]])
# x = ts.defocus(x, pa, pr)
print(pa.shape, pr.shape, x.shape)

print(f.shape)

x = x.permute((0, 3, 1, 2))
f = f.permute((0, 3, 1, 2))
print(x.shape, "======")

model = th.nn.Sequential(
    th.nn.Conv2d(2, 32, 3, 1, 1),
    th.nn.InstanceNorm2d(64),
    th.nn.LeakyReLU(),
    th.nn.Conv2d(32, 64, 3, 1, 1),
    th.nn.InstanceNorm2d(64),
    th.nn.LeakyReLU(),
    th.nn.Conv2d(64, 2, 3, 1, 1),
)


lossfdafn = ts.FourierAmplitudeLoss(mode='mae', axis=(2, 3), norm=True, reduction='mean')
lossfdafn = ts.CMSELoss(cdim=-1, norm='max', reduction='mean')
lossfdafn = th.nn.MSELoss()
lossfn = tb.entropyLoss('natural', axis=(2, 3), reduction='mean')  # OK
optimizer = th.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': 1e2}], lr=1e2, weight_decay=0)


for k in range(700):
    y = model(x)
    # y = y.mean(axis=1, keepdims=True)
    optimizer.zero_grad()
    # lossfda = lossfdafn(th.view_as_complex(y), th.view_as_complex(f))
    lossfda = lossfdafn(y, f)
    lossent = lossfn(y)
    loss = lossfda
    # loss = lossfda
    loss.backward()
    optimizer.step()

    print(loss.item(), lossfda.item(), lossent.item())

x = x.permute(0, 2, 3, 1)
y = y.permute(0, 2, 3, 1)
f = f.permute(0, 2, 3, 1)
x = x.squeeze(0).detach()
y = y.squeeze(0).detach()
f = f.squeeze(0).detach()

xm = tb.mapping(x.pow(2).sum(-1).sqrt())
ym = tb.mapping(y.pow(2).sum(-1).sqrt())
fm = tb.mapping(f.pow(2).sum(-1).sqrt())

plt.figure()
plt.subplot(131)
plt.imshow(xm)
plt.subplot(132)
plt.imshow(ym)
plt.subplot(133)
plt.imshow(fm)
plt.show()


# plt.figure()
# plt.subplot(231)
# plt.imshow(x[..., 0])
# plt.subplot(232)
# plt.imshow(x[..., 1])
# plt.subplot(233)
# plt.imshow(np.arctan(x[..., 1]) / x[..., 0])

# plt.subplot(234)
# plt.imshow(y[..., 0])
# plt.subplot(235)
# plt.imshow(y[..., 1])
# plt.subplot(236)
# plt.imshow(np.arctan(y[..., 1]) / y[..., 0])
# plt.show()

