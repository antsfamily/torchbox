import numpy as np
import torch as th
import torchbox as tb

n = 10
print(np.fft.fftfreq(n, d=0.1), 'numpy')
print(th.fft.fftfreq(n, d=0.1), 'torch')
print(tb.fftfreq(n, fs=10., norm=False), 'fftfreq, norm=False, shift=False')
print(tb.fftfreq(n, fs=10., norm=True), 'fftfreq, norm=True, shift=False')
print(tb.fftfreq(n, fs=10., shift=True), 'fftfreq, norm=False, shift=True')
print(tb.freq(n, fs=10., norm=False), 'freq, norm=False, shift=False')
print(tb.freq(n, fs=10., norm=True), 'freq, norm=True, shift=False')
print(tb.freq(n, fs=10., shift=True), 'freq, norm=False, shift=True')

print('-------------------')

n = 11
print(np.fft.fftfreq(n, d=0.1), 'numpy')
print(th.fft.fftfreq(n, d=0.1), 'torch')
print(tb.fftfreq(n, fs=10., norm=False), 'fftfreq, norm=False, shift=False')
print(tb.fftfreq(n, fs=10., norm=True), 'fftfreq, norm=True, shift=False')
print(tb.fftfreq(n, fs=10., shift=True), 'fftfreq, norm=False, shift=True')
print(tb.freq(n, fs=10., norm=False), 'freq, norm=False, shift=False')
print(tb.freq(n, fs=10., norm=True), 'freq, norm=True, shift=False')
print(tb.freq(n, fs=10., shift=True), 'freq, norm=False, shift=True')
