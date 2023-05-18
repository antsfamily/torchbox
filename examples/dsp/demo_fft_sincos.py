#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-06 10:38:13
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchbox as tb
import matplotlib.pyplot as plt

shift = True
frq = [10, 10]
amp = [0.8, 0.6]
Fs = 80
Ts = 2.
Ns = int(Fs * Ts)

t = th.linspace(-Ts / 2., Ts / 2., Ns).reshape(Ns, 1)
f = tb.freq(Ns, Fs, shift=shift)
f = tb.fftfreq(Ns, Fs, norm=False, shift=shift)

# ---complex vector in real representation format
x = amp[0] * th.cos(2. * th.pi * frq[0] * t) + 1j * amp[1] * th.sin(2. * th.pi * frq[1] * t)

# ---do fft
Xc = tb.fft(x, n=Ns, cdim=None, dim=0, keepdim=False, shift=shift)

# ~~~get real and imaginary part
xreal = tb.real(x, cdim=None, keepdim=False)
ximag = tb.imag(x, cdim=None, keepdim=False)
Xreal = tb.real(Xc, cdim=None, keepdim=False)
Ximag = tb.imag(Xc, cdim=None, keepdim=False)

# ---do ifft
x̂ = tb.ifft(Xc, n=Ns, cdim=None, dim=0, keepdim=False, shift=shift)
 
# ~~~get real and imaginary part
x̂real = tb.real(x̂, cdim=None, keepdim=False)
x̂imag = tb.imag(x̂, cdim=None, keepdim=False)

plt.figure()
plt.subplot(131)
plt.grid()
plt.plot(t, xreal)
plt.plot(t, ximag)
plt.legend(['real', 'imag'])
plt.title('signal in time domain')
plt.subplot(132)
plt.grid()
plt.plot(f, Xreal)
plt.plot(f, Ximag)
plt.legend(['real', 'imag'])
plt.title('signal in frequency domain')
plt.subplot(133)
plt.grid()
plt.plot(t, x̂real)
plt.plot(t, x̂imag)
plt.legend(['real', 'imag'])
plt.title('reconstructed signal')
plt.show()

# ---complex vector in real representation format
x = tb.c2r(x, cdim=-1)

# ---do fft
Xc = tb.fft(x, n=Ns, cdim=-1, dim=0, keepdim=False, shift=shift)

# ~~~get real and imaginary part
xreal = tb.real(x, cdim=-1, keepdim=False)
ximag = tb.imag(x, cdim=-1, keepdim=False)
Xreal = tb.real(Xc, cdim=-1, keepdim=False)
Ximag = tb.imag(Xc, cdim=-1, keepdim=False)

# ---do ifft
x̂ = tb.ifft(Xc, n=Ns, cdim=-1, dim=0, keepdim=False, shift=shift)
 
# ~~~get real and imaginary part
x̂real = tb.real(x̂, cdim=-1, keepdim=False)
x̂imag = tb.imag(x̂, cdim=-1, keepdim=False)

plt.figure()
plt.subplot(131)
plt.grid()
plt.plot(t, xreal)
plt.plot(t, ximag)
plt.legend(['real', 'imag'])
plt.title('signal in time domain')
plt.subplot(132)
plt.grid()
plt.plot(f, Xreal)
plt.plot(f, Ximag)
plt.legend(['real', 'imag'])
plt.title('signal in frequency domain')
plt.subplot(133)
plt.grid()
plt.plot(t, x̂real)
plt.plot(t, x̂imag)
plt.legend(['real', 'imag'])
plt.title('reconstructed signal')
plt.show()
