
import numpy as np
import torch as th
import torchbox as tb
import matplotlib.pyplot as plt

Ts = 2.
Ns = 100
tr = np.linspace(0, Ts, Ns)

Di = 1.3
tri = tr / Di

x = np.sin(2 * np.pi * 10 * tr)
xi_np = np.interp(tri, tr, x)


thinterpfn = tb.Interp1()
tr_th = th.tensor(tr)
tri_th = th.tensor(tri)
x_th = th.tensor(x)
xi_th = thinterpfn(tr_th, x_th, tri_th)
xi_th = xi_th.cpu().numpy().squeeze()


plt.figure()
plt.plot(tr, x, '-b')
plt.plot(tri, xi_np, '-go')
plt.plot(tri, xi_th, '-r+')
plt.show()


N = 3
trs_th = tr_th.repeat(N, 1)
xs_th = x_th.repeat(N, 1)
tris_th = tri_th.repeat(N, 1)

xis_th = thinterpfn(trs_th, xs_th, tris_th)
xis_th = xis_th.cpu().numpy().squeeze()

plt.figure()
plt.subplot(131)
plt.plot(tr, x, '-b')
plt.plot(tri, xi_np, '-go')
plt.plot(tri, xis_th[0], '-r+')
plt.subplot(132)
plt.plot(tr, x, '-b')
plt.plot(tri, xi_np, '-go')
plt.plot(tri, xis_th[1], '-r+')
plt.subplot(133)
plt.plot(tr, x, '-b')
plt.plot(tri, xi_np, '-go')
plt.plot(tri, xis_th[2], '-r+')
plt.show()
