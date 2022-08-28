import numpy as np
import torch as th
import matplotlib.pyplot as plt

PI = np.pi
f0 = 100
Fs = 1000
Ts = 0.1
Ns = int(Fs * Ts)

f = np.linspace(0., Fs, Ns)
t = np.linspace(0, Ts, Ns)
x_np = np.cos(2. * PI * f0 * t) + 1j * np.sin(2. * PI * f0 * t)

device = th.device('cuda:0')
# x_th = th.tensor(x_np, dtype=th.complex64)
x_th = th.tensor([x_np.real, x_np.imag], dtype=th.float32).transpose(1, 0)
x_th = x_th.to(device)
print(x_th.shape, type(x_th))
# print(x_th)

y1 = np.fft.fft(x_np, Ns)
y1 = np.abs(y1)

y2 = th.fft.fft(x_th, signal_ndim=1)
y2 = th.abs(y2[:, 0] + 1j * y2[:, 1]).cpu()

print(np.sum(y1 - y2.numpy()))

plt.figure()
plt.subplot(131)
plt.plot(t, x_np)
plt.grid()
plt.subplot(132)
plt.plot(f, y1)
plt.grid()
plt.subplot(133)
plt.plot(f, y2)
plt.grid()
plt.show()


x_ths = th.tensor([x_th.cpu().numpy(), x_th.cpu().numpy(), x_th.cpu().numpy()], dtype=th.float32)

print(x_ths.shape)

ys = th.fft.fft(x_ths, signal_ndim=1)
ys = th.abs(ys[:, :, 0] + 1j * ys[:, :, 1]).cpu()

plt.figure()
plt.subplot(131)
plt.plot(f, ys[0])
plt.grid()
plt.subplot(132)
plt.plot(f, ys[1])
plt.grid()
plt.subplot(133)
plt.plot(f, ys[2])
plt.grid()
plt.show()

