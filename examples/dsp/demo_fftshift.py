import numpy as np
import torchbox as tb
import torch as th


x = [1, 2, 3, 4, 5, 6]
y = np.fft.fftshift(x)
print(y)
x = th.tensor(x)
y = tb.fftshift(x)
print(y)

x = [1, 2, 3, 4, 5, 6, 7]
y = np.fft.fftshift(x)
print(y)
x = th.tensor(x)
y = tb.fftshift(x)
print(y)


x = [1, 2, 3, 4, 5, 6]
y = np.fft.ifftshift(x)
print(y)
x = th.tensor(x)
y = tb.ifftshift(x)
print(y)

x = [1, 2, 3, 4, 5, 6, 7]
y = np.fft.ifftshift(x)
print(y)
x = th.tensor(x)
y = tb.ifftshift(x)
print(y)


# shift all axes

axis = (0, 1)
# axis = 0
axis = 1

x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
y = np.fft.fftshift(x, axis)
print(y)
x = th.tensor(x)
y = tb.fftshift(x, axis)
print(y)


x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
y = np.fft.fftshift(x, axis)
print(y)
x = th.tensor(x)
y = tb.fftshift(x, axis)
print(y)


x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
y = np.fft.ifftshift(x, axis)
print(y)
x = th.tensor(x)
y = tb.ifftshift(x, axis)
print(y)


x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
y = np.fft.ifftshift(x, axis)
print(y)
x = th.tensor(x)
y = tb.ifftshift(x, axis)
print(y)
