class FourierLoss(th.nn.Module):
    r"""Fourier Domain Loss

    Compute loss in Fourier domain. Given input :math:`{\bm P}`, target :math:`\bm G`, 
    
    .. math::
       L = \varepsilon({\mathcal F}({\bm P}), {\mathcal F}({\bm G}))
    
    where, :math:`{\bm P}`, :math:`\bm G` can be real-valued and complex-valued data, :math:`\varepsilon(\cdot)` is a
    function, such as mean square error, absolute error, ...

    Parameters
    ----------
    err : str, object, optional
        string type will be converted to function by :func:`eval`, such as ``'th.nn.MSELoss()'`` (default), 
        ``'tb.SSELoss(cdim=None, dim=(-2, -1), reduction=None)'``, ``'tb.CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)'``, ...
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


    please see also :func:`th.nn.fft.fft` and :func:`th.nn.fft.ifft`.

    Examples
    ---------

    Compute loss of data in real and complex representation, respectively.

    ::

        th.manual_seed(2020)
        xr = th.randn(10, 2, 4, 4) * 100
        yr = th.randn(10, 2, 4, 4) * 100
        xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
        yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

        errr = "tb.SSELoss(cdim=1, dim=(-2, -1), reduction='mean')"
        err = "tb.SSELoss(cdim=None, dim=(-2, -1), reduction='mean')"
        # err = 'th.nn.MSELoss()'

        flossr = FourierLoss(err=errr, cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None)
        flossc = FourierLoss(err=err, cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None)
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        flossr = FourierLoss(err=errr, cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None)
        flossc = FourierLoss(err=err, cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None)
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        # ---output
        tensor(2325792.)
        tensor(2325792.)
        tensor(145362.)
        tensor(145362.)

    """

    def __init__(self, err='th.nn.MSELoss()', cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """   

class FourierAmplitudeLoss(th.nn.Module):
    r"""Fourier Domain Amplitude Loss

    compute amplitude loss in fourier domain.

    Parameters
    ----------
    err : str, object, optional
        string type will be converted to function by :func:`eval`, such as ``'th.nn.MSELoss()'`` (default), 
        ``'tb.SSELoss(cdim=None, dim=(-2, -1), reduction=None)'``, ``'tb.CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)'``, ...
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

    please see also :func:`th.nn.fft.fft` and :func:`th.nn.fft.ifft`.

    Examples
    ---------

    Compute loss of data in real and complex representation, respectively.

    ::

        th.manual_seed(2020)
        xr = th.randn(10, 2, 4, 4) * 100
        yr = th.randn(10, 2, 4, 4) * 100
        xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
        yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

        errr = "tb.SSELoss(cdim=1, dim=(-2, -1), reduction='mean')"
        err = "tb.SSELoss(cdim=None, dim=(-2, -1), reduction='mean')"
        # err = 'th.nn.MSELoss()'

        flossr = FourierAmplitudeLoss(err=err, cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None)
        flossc = FourierAmplitudeLoss(err=err, cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None)
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        flossr = FourierAmplitudeLoss(err=err, cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None)
        flossc = FourierAmplitudeLoss(err=err, cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None)
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        # ---output
        tensor(456761.5625)
        tensor(456761.5625)
        tensor(28547.5977)
        tensor(28547.5977)

    """

    def __init__(self, err='th.nn.MSELoss()', cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """   

class FourierPhaseLoss(th.nn.Module):
    r"""Fourier Domain Phase Loss

    compute phase loss in fourier domain.

    Parameters
    ----------
    err : str, object, optional
        string type will be converted to function by :func:`eval`, such as ``'th.nn.MSELoss()'`` (default), 
        ``'tb.SSELoss(cdim=None, dim=(-2, -1), reduction=None)'``, ``'tb.CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)'``, ...
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


    please see also :func:`th.nn.fft.fft` and :func:`th.nn.fft.ifft`.

    Examples
    ---------

    Compute loss of data in real and complex representation, respectively.

    ::

        th.manual_seed(2020)
        xr = th.randn(10, 2, 4, 4) * 100
        yr = th.randn(10, 2, 4, 4) * 100
        xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
        yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

        errr = "tb.SSELoss(cdim=1, dim=(-2, -1), reduction='mean')"
        err = "tb.SSELoss(cdim=None, dim=(-2, -1), reduction='mean')"
        # err = 'th.nn.MSELoss()'

        flossr = FourierPhaseLoss(err=err, cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None)
        flossc = FourierPhaseLoss(err=err, cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None)
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        flossr = FourierPhaseLoss(err=err, cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None)
        flossc = FourierPhaseLoss(err=err, cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None)
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        # ---output
        tensor(106.8749)
        tensor(106.8749)
        tensor(106.8749)
        tensor(106.8749)

    """

    def __init__(self, err='th.nn.MSELoss()', cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """   

class FourierNormLoss(th.nn.Module):
    r"""FourierNormLoss

    .. math::
       C = \frac{{\rm E}(|I|^2)}{[E(|I|)]^2}

    see Fast Fourier domain optimization using hybrid

    """

    def __init__(self, reduction='mean', p=1.5):
        ...

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


