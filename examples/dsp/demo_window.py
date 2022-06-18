#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-18 10:14:12
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torchbox as tb
import matplotlib.pyplot as plt

n = 512
wtypebartlett = 'bartlett'
wtypeblackman = 'blackman'
wtypehamming = 'hamming 0.54 0.46'
wtypehanning = 'hanning'
wtypekaiser = 'kaiser 12'
wbartlett = tb.window(n, wtype=wtypebartlett)
wblackman = tb.window(n, wtype=wtypeblackman)
whamming = tb.window(n, wtype=wtypehamming)
whanning = tb.window(n, wtype=wtypehanning)
wkaiser = tb.window(n, wtype=wtypekaiser)

plt.figure()
plt.grid()
plt.plot(wbartlett)
plt.plot(wblackman)
plt.plot(whamming)
plt.plot(whanning)
plt.plot(wkaiser)
plt.legend([wtypebartlett, wtypeblackman, wtypehamming, wtypehanning, wtypekaiser])
plt.show()



