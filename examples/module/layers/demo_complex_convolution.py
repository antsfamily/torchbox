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

datafile = 'data/sarimg/ALPSRP020160970_Vr7180_R3_FocusedEach_5x256x256_AutoFocusPolyPhiMinEntropy_Epoch1000.h5'
data = tb.loadh5(datafile)
SI, ca, cr = data['SI'], data['ca'], data['cr']
N, Na, Nr, Nc = SI.shape
print(SI.shape)
x = th.from_numpy(SI[0]).unsqueeze(0).unsqueeze(0)
print(x.shape)
t = x.clone()

carange = [[200, 2], [400, 10]]

crrange = [[-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]]

ppeg = ts.PolyPhaseErrorGenerator(carange, crrange, seed)

ca, cr = ppeg.mkpec(n=1, seed=None)

xa = ts.ppeaxis(Na, norm=True, shift=ftshift, mode='fftfreq')

xr = ts.ppeaxis(Nr, norm=True, shift=ftshift, mode='fftfreq')

pa, pr = ts.polype(ca, xa), ts.polype(cr, xr)

print(pa.shape, pr.shape, x.shape)
x = ts.defocus(x, pa, pr)
print(pa.shape, pr.shape, x.shape)

print(x.shape)

plt.figure()
plt.subplot(221)
plt.imshow(x[0, 0, ...].pow(2).sum(-1).sqrt())
plt.subplot(222)
plt.grid()
plt.plot(pa.squeeze(0))
plt.subplot(223)
plt.imshow(x[0, 0, :, :, 0])
plt.subplot(224)
plt.imshow(x[0, 0, :, :, 1])
plt.show()


class ConvNet(th.nn.Module):

    def __init__(self):
        super().__init__()
        self.convf1 = ts.ComplexConv2d(1, 1, 3, 1, 1)
        self.convf2 = ts.ComplexConv2d(1, 1, 3, 1, 1)
        self.convd1 = ts.ComplexConv2d(1, 1, 3, 1, 1)
        self.convd2 = ts.ComplexConv2d(1, 1, 3, 1, 1)

    def forward(self, x):
        y = self.convf1(x)
        y = self.convf2(y)
        x = self.convd1(y)
        x = self.convd2(x)
        return y, x


net = ConvNet()

loss_fda_fn = ts.FourierAmplitudeLoss(mode='mae', axis=(2, 3), norm=True, reduction='mean')
loss_ent_fn = tb.entropyLoss('natural', axis=(2, 3), reduction='mean')  # OK
loss_mae_fn = ts.CMAELoss(reduction='mean', norm=False)
optimizer = th.optim.AdamW([{'params': filter(lambda p: p.requires_grad, net.parameters()), 'initial_lr': 1e-1}], lr=1e2, weight_decay=1e-2)


for k in range(400):
    optimizer.zero_grad()
    y, px = net.forward(x)
    lossfda = loss_fda_fn(th.view_as_complex(y), th.view_as_complex(x))
    lossent = loss_ent_fn(y)
    lossmae = loss_mae_fn(px, x)
    # loss = lossmae
    loss = lossent + lossmae
    # loss = lossent + 0.01 * lossfda + lossmae
    loss.backward()
    optimizer.step()

    print(loss.item(), lossfda.item(), lossent.item(), lossmae.item())

x = x.squeeze(0).squeeze(0).detach().numpy()
y = y.squeeze(0).squeeze(0).detach().numpy()

print(np.sum(x - y))


plt.figure()
plt.subplot(121)
plt.imshow(np.sqrt(np.sum(x**2, -1)))
plt.subplot(122)
plt.imshow(np.sqrt(np.sum(y**2, -1)))
plt.show()


plt.figure()
plt.subplot(231)
plt.imshow(x[..., 0])
plt.subplot(232)
plt.imshow(x[..., 1])
plt.subplot(233)
plt.imshow(np.arctan(x[..., 1]) / x[..., 0])

plt.subplot(234)
plt.imshow(y[..., 0])
plt.subplot(235)
plt.imshow(y[..., 1])
plt.subplot(236)
plt.imshow(np.arctan(y[..., 1]) / y[..., 0])
plt.show()

