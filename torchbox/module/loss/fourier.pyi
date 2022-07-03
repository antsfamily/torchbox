    """

    def __init__(self, err='mse', cdim=None, ftdim=(-2, -1), iftdim=None, keepcdim=False, ftn=None, iftn=None, ftnorm=None, iftnorm=None, reduction='mean'):
    def forward(self, P, G):
    r"""Fourier Domain Amplitude Loss

    compute amplitude loss in fourier domain.

    Parameters
    ----------
    err : str, loss function, optional
        ``'MSE'``, ``'MAE'`` or torch's loss function, by default ``'mse'``.
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to None, defaults is None.
        For example, :math:`{\bm X}_c\in {\mathbb C}^{N\times C\times H\times W}` is
        represented as a real-valued tensor :math:`{\bm X}_r\in {\mathbb R}^{N\times C\times H\times W\ times 2}`,
        then :attr:`cdim` equals to -1 or 4.
    ftdim : tuple, None, optional
        the dimensions for Fourier transformation. by default (-2, -1).
    iftdim : tuple, None, optional
        the dimension for inverse Fourier transformation, by default None.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    ftn : int, None, optional
        the number of points for Fourier transformation, by default None
    iftn : int, None, optional
        the number of points for inverse Fourier transformation, by default None
    ftnorm : str, None, optional
        the normalization method for Fourier transformation, by default None
        - "forward" - normalize by 1/n
        - "backward" - no normalization
        - "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
    iftnorm : str, None, optional
        the normalization method for inverse Fourier transformation, by default None
        - "forward" - no normalization
        - "backward" - normalize by 1/n
        - "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
    reduction : str, optional
        reduction behavior, ``'sum'`` or ``'mean'``, by default ``'mean'``

    please see :func:`th.nn.fft.fft` and :func:`th.nn.fft.ifft`.

    Examples
    ---------

    Compute loss of data in real and complex representation, respectively.

    ::

        th.manual_seed(2020)
        xr = th.randn(10, 2, 4, 4) * 10000
        yr = th.randn(10, 2, 4, 4) * 10000
        xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
        yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

        flossr = FourierAmplitudeLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        flossc = FourierAmplitudeLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        flossr = FourierAmplitudeLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        flossc = FourierAmplitudeLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        # ---output
        tensor(2.8548e+08)
        tensor(2.8548e+08)
        tensor(17842250.)
        tensor(17842250.)

    """

    def __init__(self, err='mse', cdim=None, ftdim=(-2, -1), iftdim=None, keepcdim=False, ftn=None, iftn=None, ftnorm=None, iftnorm=None, reduction='mean'):
    def forward(self, P, G):
    r"""Fourier Domain Phase Loss

    compute phase loss in fourier domain.

    Parameters
    ----------
    err : str, loss function, optional
        ``'MSE'``, ``'MAE'`` or torch's loss function, by default ``'mse'``
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to None, defaults is None.
        For example, :math:`{\bm X}_c\in {\mathbb C}^{N\times C\times H\times W}` is
        represented as a real-valued tensor :math:`{\bm X}_r\in {\mathbb R}^{N\times C\times H\times W\ times 2}`,
        then :attr:`cdim` equals to -1 or 4.
    ftdim : tuple, None, optional
        the dimensions for Fourier transformation. by default (-2, -1).
    iftdim : tuple, None, optional
        the dimension for inverse Fourier transformation, by default None.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    ftn : int, None, optional
        the number of points for Fourier transformation, by default None
    iftn : int, None, optional
        the number of points for inverse Fourier transformation, by default None
    ftnorm : str, None, optional
        the normalization method for Fourier transformation, by default None
        - "forward" - normalize by 1/n
        - "backward" - no normalization
        - "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
    iftnorm : str, None, optional
        the normalization method for inverse Fourier transformation, by default None
        - "forward" - no normalization
        - "backward" - normalize by 1/n
        - "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
    reduction : str, optional
        reduction behavior, ``'sum'`` or ``'mean'``, by default ``'mean'``

    please see :func:`th.nn.fft.fft` and :func:`th.nn.fft.ifft`.

    Examples
    ---------

    Compute loss of data in real and complex representation, respectively.

    ::

        th.manual_seed(2020)
        xr = th.randn(10, 2, 4, 4) * 10000
        yr = th.randn(10, 2, 4, 4) * 10000
        xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
        yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

        flossr = FourierPhaseLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        flossc = FourierPhaseLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        flossr = FourierPhaseLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        flossc = FourierPhaseLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        # ---output
        tensor(6.6797)
        tensor(6.6797)
        tensor(6.6797)
        tensor(6.6797)
    """

    def __init__(self, err='mse', cdim=None, ftdim=(-2, -1), iftdim=None, keepcdim=False, ftn=None, iftn=None, ftnorm=None, iftnorm=None, reduction='mean'):
    def forward(self, P, G):
    r"""FourierNormLoss

    .. math::
        C = \frac{{\rm E}(|I|^2)}{[E(|I|)]^2}

    see Fast Fourier domain optimization using hybrid

    """

    def __init__(self, reduction='mean', p=1.5):
    def forward(self, X, w=None):
        r"""[summary]

        [description]

        Parameters
        ----------
        X : Tensor
            After fft in azimuth
        w : Tensor, optional
            weight

        Returns
        -------
        float
            loss
        """

