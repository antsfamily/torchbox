def cosine(t, alpha=1., freq=1., phi=0., reduction=None):
    """generates cosine signal

    Parameters
    ----------
    t : Tensor
        the discretized time steps
    alpha : float, tensor or list, optional
        amplitudes, by default 1.
    freq : float, tensor or list, optional
        frequencies, by default 1.
    phi : float, tensor or list, optional
        initial phases, by default 0.
    reduction : str or None, optional
        ``'mean'``, ``'sum'`` or None
    """

class CosineTask: def __init__(self, ntask=10, arange=[0.1, 50.], frange=[3., 10.], prange=[0, 3.1415926], trange=[-0.5, 0.5, 500], tmode='uniform', seed=None) -> None:
                      ...

    def __init__(self, ntask=10, arange=[0.1, 50.], frange=[3., 10.], prange=[0, 3.1415926], trange=[-0.5, 0.5, 500], tmode='uniform', seed=None) -> None: self.ntask = ntask
        ...

    def mktask(self, ntask=10, arange=None, frange=None, prange=None, trange=None, seed=None, device=None, rett=False):
        ...


