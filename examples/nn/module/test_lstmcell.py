import torch as th

nin, nhi, nts, nbs = 2, 3, 6, 5
rnn = th.nn.LSTM(nin, nhi, 2)
rnncell0 = th.nn.LSTMCell(nin, nhi)
rnncell0.weight_ih = rnn.weight_ih_l0
rnncell0.weight_hh = rnn.weight_hh_l0
rnncell0.bias_ih = rnn.bias_ih_l0
rnncell0.bias_hh = rnn.bias_hh_l0
rnncell1 = th.nn.LSTMCell(nhi, nhi)
rnncell1.weight_ih = rnn.weight_ih_l1
rnncell1.weight_hh = rnn.weight_hh_l1
rnncell1.bias_ih = rnn.bias_ih_l1
rnncell1.bias_hh = rnn.bias_hh_l1

x = th.rand(nts, nbs, nin)
h0, c0 = th.randn(2, nbs, nhi), th.randn(2, nbs, nhi)
print(x.shape, x.sum())

y, (hx, cx) = rnn(x, (h0, c0))
print(x.shape, x.sum())
print(y.shape, hx.shape, cx.shape)
print(y.sum().item(), hx.sum().item(), cx.sum().item())

hx0, cx0 = h0[0].clone(), c0[0].clone()
hx1, cx1 = h0[1].clone(), c0[1].clone()
ycell = []
for t in range(x.shape[0]):
    hx0, cx0 = rnncell0(x[t, :, :], (hx0, cx0))
    hx1, cx1 = rnncell1(hx0, (hx1, cx1))
    ycell.append(hx1)
hxcell = th.stack((hx0, hx1), dim=0)
cxcell = th.stack((cx0, cx1), dim=0)

ycell = th.stack(ycell, dim=0)

print(ycell.shape, hxcell.shape, cxcell.shape)
print(ycell.sum().item(), hxcell.sum().item(), cxcell.sum().item())

