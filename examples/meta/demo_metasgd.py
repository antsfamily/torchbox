
import torch as th
import torchbox as tb





x = th.rand(100, 5, 20, 6, 64, 64)
y = th.rand(100, 5, 20, 2, 64, 64)


dltrain = tb.MetaDataLoader(x, y, bstask=32, nway=5, kshot=10, kquery=6, sfltask=True, sflpoint=True, dsname='train')

print(len(dltrain))
for b in range(len(dltrain)):
    xspt, yspt, xqry, yqry = dltrain.next()
    print(b, xspt.shape, yspt.shape, xqry.shape, yqry.shape)


