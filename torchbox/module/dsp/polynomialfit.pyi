class PolyFit(th.nn.Module):
    r"""Polynominal fitting

    We fit the data using a polynomial function of the form

    .. math::
       y(x, {\mathbf w})=w_{0}+w_{1} x+w_{2} x^{2}+, \cdots,+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}

    Parameters
    ----------
    w : Tensor, optional
        initial coefficient, by default None (generate randomly)
    deg : int, optional
        degree of the Polynominal, by default 1
    trainable : bool, optional
        is ``self.w`` trainable, by default True

    Examples
    --------

    ::

        th.manual_seed(2020)
        Ns, k, b = 100, 1.2, 3.0
        x = th.linspace(0, 1, Ns)
        t = x * k + b + th.randn(Ns)

        deg = (0, 1)

        polyfit = PolyFit(deg=deg)

        lossfunc = th.nn.MSELoss('mean')
        optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, polyfit.parameters()), lr=1e-1)

        for n in range(100):
            y = polyfit(x)

            loss = lossfunc(y, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("---Loss %.4f, %.4f, %.4f" % (loss.item(), polyfit.w[0], polyfit.w[1]))

        # output
        ---Loss 16.7143, -0.2315, -0.1427
        ---Loss 15.5265, -0.1316, -0.0429
        ---Loss 14.3867, -0.0319, 0.0568
        ---Loss 13.2957, 0.0675, 0.1561
        ---Loss 12.2543, 0.1664, 0.2551
                        ...
        ---Loss 0.9669, 2.4470, 1.9995
        ---Loss 0.9664, 2.4515, 1.9967
        ---Loss 0.9659, 2.4560, 1.9938
    """

    def __init__(self, w=None, deg=1, trainable=True):
        ...

    def forward(self, x):
        ...


