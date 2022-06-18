import time
import numpy as np
import torch as th
import torchbox as tb

PI = np.pi
f0 = 100
Fs = 1000
Ts = 0.1
Ns = int(Fs * Ts)

nSamples = 256

T = th.zeros(nSamples, Ns)
for n in range(nSamples):
    To = tb.randperm(-Ns, Ns, 1)[0] / Fs
    T[n, :] = th.linspace(To, Ts + To, Ns)


def fftexpnet(f, T):
    # X = th.cos(2 * np.pi * f * T) + 1j * th.sin(2 * np.pi * f * T)
    # X = th.cos(2 * np.pi * f * T)
    # X = th.view_as_real(X)
    # print(X.shape)

    Xr = th.cos(2 * np.pi * f * T)
    Xi = th.sin(2 * np.pi * f * T)
    X = th.stack([Xr, Xi], dim=-1)
    P = tb.fft(X, n=None, dim=0, norm="backward", shift=False)
    return P


# Y = th.zeros(nSamples, Ns, 2)

P = fftexpnet(f0, T)

device, dtype = th.device('cuda:0'), th.float32

num_epochs = 400
batch_size = 8

f = th.randn(1, device=device, dtype=dtype, requires_grad=True)
f = th.ones(1, device=device, dtype=dtype, requires_grad=True)
f = th.tensor(98., device=device, dtype=dtype, requires_grad=True)
lr = 1e-3

for k in range(num_epochs):
    idx = tb.randperm(0, nSamples, nSamples)
    T, P = T[idx], P[idx]

    tstart = time.time()
    for b in range(batch_size):
        bT, bP = T[b * batch_size:(b + 1) * batch_size], P[b * batch_size:(b + 1) * batch_size]
        bT, bP = bT.to(device), bP.to(device)

        bY = fftexpnet(f, bT)

        loss = (bP - bY).pow(2).sum()

        loss.backward()

        with th.no_grad():
            # print(f, f.grad)
            f -= lr * f.grad
            # Manually zero the gradients after updating weights
            f.grad.zero_()

    tend = time.time()
    print("--->epoch: %d, loss: %.4f, frequency: %.4f, time: %ss" %
          (k, loss.item(), f, tend - tstart))
