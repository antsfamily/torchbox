def cutfftcorr1(y, nfft, Nx, Nh, shape='same', dim=0, ftshift=False):
    r"""Throwaway boundary elements to get correlation results.

    Throwaway boundary elements to get correlation results.

    Parameters
    ----------
    y : tensor
        array after ``iff``.
    nfft : int
        number of fft points.
    Nx : int
        signal length
    Nh : int
        filter length
    shape : dstr
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid correlation output``
        3. ``'full' --> full correlation output``, :math:`N_x+N_h-1`
        (the default is 'same')
    dim : int
        correlation dim (the default is 0)
    ftshift : bool
        whether to shift the frequencies (the default is False)

    Returns
    -------
    y : tensor
        array with shape specified by :attr:`same`.
    """

def fftcorr1(x, h, shape='same', nfft=None, ftshift=False, eps=None, **kwargs):
    """Correlation using Fast Fourier Transformation

    Correlation using Fast Fourier Transformation.

    Parameters
    ----------
    x : tensor
        data to be convolved.
    h : tensor
        filter array, it will be expanded to the same dimensions of :attr:`x` first.
    shape : dstr, optional
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid correlation output``
        3. ``'full' --> full correlation output``, :math:`N_x+N_h-1`
        (the default is 'same')
    cdim : int or None
        If :attr:`x` is complex-valued, :attr:`cdim` is ignored. If :attr:`x` is real-valued and :attr:`cdim` is integer
        then :attr:`x` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex dim;
        otherwise (None), :attr:`x` will be treated as real-valued.
    dim : int, optional
        axis of fft operation (the default is 0, which means the first dimension)
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`x` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    nfft : int, optional
        number of fft points (the default is None, :math:`2^{nextpow2(N_x+N_h-1)}`),
        note that :attr:`nfft` can not be smaller than :math:`N_x+N_h-1`.
    ftshift : bool, optional
        whether shift frequencies (the default is False)
    eps : None or float, optional
        x[abs(x)<eps] = 0 (the default is None, does nothing)

    Returns
    -------
    y : tensor
        Correlation result array.

    """

def xcorr(A, B, shape='same', dim=0):
    r"""Cross-correlation function estimates.


    Parameters
    ----------
    A : numpy array
        data1
    B : numpy array
        data2
    mod : str, optional
        - 'biased': scales the raw cross-correlation by 1/M.
        - 'unbiased': scales the raw correlation by 1/(M-abs(lags)).
        - 'coeff': normalizes the sequence so that the auto-correlations
                   at zero lag are identically 1.0.
        - 'none': no scaling (this is the default).
    """

def accc(Sr, isplot=False):
    r"""Average cross correlation coefficient

    Average cross correlation coefficient (ACCC)

    .. math::
       \overline{C(\eta)}=\sum_{\eta} s^{*}(\eta) s(\eta+\Delta \eta)

    where, :math:`\eta, \Delta \eta` are azimuth time and it's increment.


    Parameters
    ----------
    Sr : numpy array
        SAR raw signal data :math:`N_a×N_r` or range compressed data.

    Returns
    -------
    1d array
        ACCC in each range cell.
    """

