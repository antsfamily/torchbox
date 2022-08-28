import numpy as np
import torch as th
import pyaibox as pb
import matplotlib.pyplot as plt

X_np = pb.imread('../../data/images/Einstein.png').astype('float32')
X_np = X_np + 1j * X_np

device = th.device('cuda:0')
X_th = th.tensor([X_np.real, X_np.imag], dtype=th.float32).permute(1, 2, 0)
X_th = X_th.to(device)
print(X_th.shape, type(X_th))
# print(x_th)

Y1 = np.fft.fft(X_np, axis=0)
Y1 = np.fft.fft(Y1, axis=1)
# Y1 = np.fft.fft(np.fft.fft(X_np, axis=0), axis=1)
Y1 = np.abs(Y1)

Y2 = th.fft.fft(X_th.transpose(0, 1), signal_ndim=1).transpose(0, 1)
Y2 = th.fft.fft(Y2, signal_ndim=1)
# Y2 = th.fft.fft(X_th, signal_ndim=2)
Y2 = th.abs(Y2[:, :, 0] + 1j * Y2[:, :, 1]).cpu()

print(np.sum(Y1 - Y2.numpy()))
Y1 = np.log10(Y1)
Y2 = th.log10(Y2)

plt.figure()
plt.subplot(131)
plt.imshow(np.abs(X_np))
plt.subplot(132)
plt.imshow(Y1)
plt.subplot(133)
plt.imshow(Y2)
plt.show()





