import torch as th
import torchbox as tb
import matplotlib.pyplot as plt

Ns, k, b = 100, 6.2, 3.0
x = th.linspace(0, 1, Ns)
y = x * k + b + th.randn(Ns)
# x, y = x.to('cuda:1'), y.to('cuda:1')

deg = 1
deg = 2
# deg = (2, 2)

w = tb.polyfit(x, y, deg=deg)
print(w)
w = tb.polyfit(x, y, deg=deg, C=5)
print(w)
yy = tb.polyval(w, x, deg=deg)
print(yy)

print(th.sum(th.abs(y - yy)))

plt.figure()
plt.plot(x, y, 'ob')
plt.plot(x, yy, '-r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('polynomial fitting')
plt.legend(['noised data', 'fitted'])
plt.show()
