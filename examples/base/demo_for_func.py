import time
import torch as th
import torchbox as tb


Nk = 1000

X = th.rand(10, 256, 256, 2)
cdim = -1

t1 = time.time()
for k in range(Nk):
    Z = (X**2).sum(dim=cdim).sqrt()
t2 = time.time()
print(t2-t1)

# t1 = time.time()
# for k in range(Nk):
#     Y = tb.pow(X, cdim=cdim)
# t2 = time.time()
# print(t2-t1)

t1 = time.time()
for k in range(Nk):
    Y = tb.abs(X, cdim=cdim)
t2 = time.time()
print(t2-t1)

print((Y-Z).abs().sum())
