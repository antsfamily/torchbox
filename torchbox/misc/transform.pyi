def zscore(X, meanv=None, stdv=None, cdim=None, dim=None, retall=False):
    r"""standardization/zscore

    .. math::
        \bar{X} = \frac{X-\mu}{\sigma}


    Parameters
    ----------
    X : Tensor
        data to be normalized,
    meanv : list or None, optional
        mean value (the default is None, which means auto computed)
    stdv : list or None, optional
        standard deviation (the default is None, which means auto computed)
    cdim : int or None, optional
        complex dimension
    dim : list or int, optional
        specify the axis for computing mean and standard deviation (the default is None, which means all elements)
    retall : bool, optional
        if True, also return the mean and std (the default is False, which means just return the standardized data)
    """

def scale(X, st=[0, 1], sf=None, istrunc=True, retall=False):
    r"""
    Scale data.

    .. math::
        x \in [a, b] \rightarrow y \in [c, d]

    .. math::
        y = (d-c)(x-a) / (b-a) + c.

    Parameters
    ----------
    X : tensor_like
        The data to be scaled.
    st : tuple, list, optional
        Specifies the range of data after beening scaled. Default [0, 1].
    sf : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    istrunc : bool
        Specifies wether to truncate the data to [a, b], For example,
        If sf == [a, b] and 'istrunc' is true,
        then X[X < a] == a and X[X > b] == b.
    retall : bool
        If ``True``, also return :attr:`st` and :attr:`sf`.

    Returns
    -------
    out : Tensor
        Scaled data tensor.
    st, sf : list or tuple
        If :attr:`retall` is true, also be returned
    """

def quantization(X, idrange=None, odrange=[0, 31], odtype='auto', retall=False):
    r"""

    Quantize data.

    .. math::
        x \in [a, b] \rightarrow y \in [c, d]

    .. math::
        y = (d-c)(x-a) / (b-a) + c.

    Parameters
    ----------
    X : Tensor
        The data to be quantized with shape :math:`N_a×N_r ∈ {\mathbb R}`, or :math:`N_a×N_r ∈ {\mathbb C}`.
    idrange : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    odrange : tuple, list, optional
        Specifies the range of data after beening quantized. Default [0, 31].
    odtype : str, None, optional
        output data type, supportted are ``'auto'`` (auto infer, default), or torch tensor's dtype string.
        If the type of :attr:`odtype` is not string(such as None),
        the type of output data is the same with input.
    retall : bool
        If ``True``, also return :attr:`st` and :attr:`idrange`.

    Returns
    -------
    out : Tensor
        Quantized data tensor, if the input is complex, will return a tensor with shape :math:`N_a×N_r×2 ∈ {\mathbb R}`.
    idrange, odrange : list or tuple
        If :attr:`retall` is true, also be returned
    """

def db20(x):
    r"""Computes dB value of a tensor

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor (dB)
    """

def ct2rt(x, dim=0):
    r"""Converts a complex-valued tensor to a real-valued tensor

    Converts a complex-valued tensor :math:`{\bf x}` to a real-valued tensor with FFT and conjugate symmetry.


    Parameters
    ----------
    x : Tensor
        The input tensor :math:`{\bf x}`.
    dim : int
        The axis for excuting FFT.

    Returns
    -------
    Tensor
        The output tensor :math:`{\bf y}`.

    see also :func:`rt2ct`.

    Examples
    ---------

    .. image:: ./_static/CT2RTRT2CTdemo.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::


        import torchbox as tb

        datafolder = tb.data_path('optical')
        xr = tb.imread(datafolder + 'Einstein256.png')
        xi = tb.imread(datafolder + 'LenaGRAY256.png')

        x = xr + 1j * xi

        y = tb.ct2rt(x, dim=0)
        z = tb.rt2ct(y, dim=0)

        print(x.shape, y.shape, z.shape)
        print(x.dtype, y.dtype, z.dtype)
        print(x.abs().min(), x.abs().max())
        print(y.abs().min(), y.abs().max())
        print(z.abs().min(), z.abs().max())


        plt = tb.imshow([x.real, x.imag, y.real, y.imag, z.real, z.imag], nrows=3, ncols=2,
                        titles=['original(real)', 'original(imag)', 'converted(real)', 
                        'converted(imag)', 'reconstructed(real)', 'reconstructed(imag)'])
        plt.show()

    """

def rt2ct(y, dim=0):
    r"""Converts a real-valued tensor to a complex-valued tensor

    Converts a real-valued tensor :math:`{\bf y}` to a complex-valued tensor with FFT and conjugate symmetry.


    Parameters
    ----------
    y : Tensor
        The input tensor :math:`{\bf y}`.
    dim : int
        The axis for excuting FFT.

    Returns
    -------
    Tensor
        The output tensor :math:`{\bf x}`.
    
    see also :func:`ct2rt`.

    """


