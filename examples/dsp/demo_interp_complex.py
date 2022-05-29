
import numpy as np
import matplotlib.pyplot as plt

x = [1.5, 4.0]
xp = [2, 3, 5]
fp = [1.0j, 0, 2 + 3j]
f = np.interp(x, xp, fp)
print(f)

x = [1.5, 4.0]
xp = [2, 3, 5]

fp = [0, 0, 2]
f = np.interp(x, xp, fp)
print(f)

fp = [1.0, 0, 3]
f = np.interp(x, xp, fp)
print(f)


Ts = 2.
Ns = 100
tr = np.linspace(0, Ts, Ns)

Di = 1.3
tri = tr / Di

x = np.cos(2 * np.pi * 10 * tr) + 1j * np.sin(2 * np.pi * 10 * tr)
xdn = x
print(tri.shape, tr.shape, xdn.shape)
xic = np.interp(tri, tr, xdn)
xir = np.interp(tri, tr, xdn.real) + 1j * np.interp(tri, tr, xdn.imag)


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


plt.figure()
plt.plot(np.angle(xic))
plt.plot(np.angle(xir))
plt.show()
