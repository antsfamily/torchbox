
import numpy as np
import matplotlib.pyplot as plt

Ts = 2.
Ns = 100
tr = np.linspace(0, Ts, Ns)

Di = 1.3
tri = tr / Di

x = np.sin(2 * np.pi * 10 * tr) + 1j * np.cos(2 * np.pi * 10 * tr)
xic = np.interp(tri, tr, x)
xir = np.interp(tri, tr, x.real) + 1j * np.interp(tri, tr, x.imag)


print(np.sum(np.abs(xic - xir)))


plt.figure()
plt.subplot(121)
plt.plot(tr, x.real, '-b')
plt.plot(tri, xic.real, '-go')
plt.plot(tri, xir.real, '-r+')
plt.subplot(122)
plt.plot(tr, x.real, '-b')
plt.plot(tri, xic.real, '-go')
plt.plot(tri, xir.real, '-r+')
plt.show()

