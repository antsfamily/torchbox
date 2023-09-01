def snr(x, n=None, **kwargs):
    r"""computes signal-to-noise ratio

    .. math::
        {\rm SNR} = 10*{\rm log10}(\frac{P_x}{P_n})
    
    where, :math:`P_x, P_n` are the power summary of the signal and noise:

    .. math::
       P_x = \sum_{i=1}^N |x_i|^2 \\
       P_n = \sum_{i=1}^N |n_i|^2 
    
    ``snr(x, n)`` equals to matlab's ``snr(x, n)``

    Parameters
    ----------
    x : Tensor
        The pure signal data.
    n : ndarray, tensor
        The noise data.
    cdim : None or int, optional
        If :attr:`x` and :attr:`n` are complex-valued but represented in real format, 
        :attr:`cdim` or :attr:`cdim` should be specified. If not, it's set to :obj:`None`, 
        which means :attr:`x` and :attr:`n` are real-valued or complex-valued in complex format.
    dim : int or None, optional
        Specifies the dimensions for computing SNR, if not specified, it's set to :obj:`None`, 
        which means all the dimensions.
    keepdim : int or None, optional
        keep the complex dimension? (False for default)
    reduction : str or None, optional
        The reduce operation in batch dimension. Supported are ``'mean'``, ``'sum'`` or :obj:`None`.
        If not specified, it is set to :obj:`None`.
    
    Returns
    -----------
      : scalar
        The SNRs.

    Examples
    ----------

    ::

        import torch as th
        import torchbox as tb
    
        tb.setseed(seed=2020, target='torch')
        x = 10 * th.randn(5, 2, 3, 4)
        n = th.randn(5, 2, 3, 4)
        snrv = snr(x, n, cdim=1, dim=(2, 3), reduction=None)
        print(snrv)
        snrv = snr(x, n, cdim=1, dim=(2, 3), reduction='mean')
        print(snrv)
        x = tb.r2c(x, cdim=1)
        n = tb.r2c(n, cdim=1)
        snrv = snr(x, n, cdim=None, dim=(1, 2), reduction='mean')
        print(snrv)
        
        ---output
        tensor([17.5840, 20.6824, 20.5385, 18.3238, 19.4630])
        tensor(19.3183)
        tensor(19.3183)

    """

def psnr(P, G, vpeak=None, **kwargs):
    r"""Peak Signal-to-Noise Ratio

    The Peak Signal-to-Noise Ratio (PSNR) is expressed as

    .. math::
        {\rm psnrv} = 10 \log10(\frac{V_{\rm peak}^2}{\rm MSE})

    For float data, :math:`V_{\rm peak} = 1`;

    For interges, :math:`V_{\rm peak} = 2^{\rm nbits}`,
    e.g. uint8: 255, uint16: 65535 ...


    Parameters
    -----------
    P : array_like
        The data to be compared. For image, it's the reconstructed image.
    G : array_like
        Reference data array. For image, it's the original image.
    vpeak : float, int or None, optional
        The peak value. If None, computes automaticly.
    cdim : None or int, optional
        If :attr:`P` and :attr:`G` are complex-valued but represented in real format, 
        :attr:`cdim` or :attr:`cdim` should be specified. If not, it's set to :obj:`None`, 
        which means :attr:`P` and :attr:`G` are real-valued or complex-valued in complex format.
    keepdim : int or None, optional
        keep the complex dimension?
    dim : int or None, optional
        Specifies the dimensions for computing SNR, if not specified, it's set to :obj:`None`, 
        which means all the dimensions.
    reduction : str or None, optional
        The reduce operation in batch dimension. Supported are ``'mean'``, ``'sum'`` or :obj:`None`.
        If not specified, it is set to :obj:`None`.
    
    Returns
    -------
    psnrv : float
        Peak Signal to Noise Ratio value.

    Examples
    ---------

    ::

        import torch as th
        import torchbox as tb
    
        print('---psnr')
        tb.setseed(seed=2020, target='torch')
        P = 255. * th.rand(5, 2, 3, 4)
        G = 255. * th.rand(5, 2, 3, 4)
        snrv = psnr(P, G, vpeak=None, cdim=1, dim=(2, 3), reduction=None)
        print(snrv)
        snrv = psnr(P, G, vpeak=None, cdim=1, dim=(2, 3), reduction='mean')
        print(snrv)
        P = tb.r2c(P, cdim=1, keepdim=False)
        G = tb.r2c(G, cdim=1, keepdim=False)
        snrv = psnr(P, G, vpeak=255, cdim=None, dim=(1, 2), reduction='mean')
        print(snrv)

        # ---output
        tensor([4.4584, 5.0394, 5.1494, 3.6585, 4.6466])
        tensor(4.5905)
        tensor(4.5905)

    """


