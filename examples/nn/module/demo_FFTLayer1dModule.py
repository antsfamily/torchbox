import time
import numpy as np
import torch as th
import pyaibox as pb
from fftnet1d import FFTNet1d

PI = np.pi
f0 = 100
Fs = 1000
Ts = 0.1
Ns = int(Fs * Ts)

nSamples = 100

T = th.zeros(nSamples, Ns)
for n in range(nSamples):
    To = pb.randperm(-Ns, Ns, 1)[0] / Fs
    T[n, :] = th.linspace(To, Ts + To, Ns)

# Y = th.zeros(nSamples, Ns, 2)
X = th.cos(2 * np.pi * f0 * T) + 1j * th.sin(2 * np.pi * f0 * T)
X = th.view_as_real(X)

P = th.fft.fft(X, signal_ndim=1, normalized=False)

device = th.device('cuda:0')
# T, P = T.to(device), P.to(device)

num_epochs = 100
batch_size = 4

net = FFTNet1d()
net.to(device=device)

for param in net.parameters():
    print(param)

loss_func = th.nn.MSELoss(reduction='mean')
optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-1)
scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.999, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)

for k in range(num_epochs):
    idx = pb.randperm(0, nSamples, nSamples)
    T, P = T[idx], P[idx]

    tstart = time.time()
    for b in range(batch_size):
        bT, bP = T[b * batch_size:(b + 1) * batch_size], P[b * batch_size:(b + 1) * batch_size]
        bT, bP = bT.to(device), bP.to(device)

        bY = net.forward(bT)
        loss = loss_func(bY, bP)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        scheduler.step()

    tend = time.time()

    print("--->epoch: %d, loss: %.4f, frequency: %.4f, time: %ss" %
          (k, loss.item(), net.f, tend - tstart))
