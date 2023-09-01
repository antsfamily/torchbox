def freq(n, fs, norm=False, shift=False, dtype=th.float32, device='cpu'):
    r"""Return the sample frequencies

    Return the sample frequencies.

    Given a window length `n` and a sample rate `fs`, if shift is ``True``::

      f = [-n/2, ..., n/2] / (d*n)

    Given a window length `n` and a sample rate `fs`, if shift is ``False``::

      f = [0, 1, ..., n] / (d*n)

    If :attr:`norm` is ``True``, :math:`d = 1`, else :math:`d = 1/f_s`.

    Parameters
    ----------
    fs : float
        Sampling rate.
    n : int
        Number of samples.
    norm : bool
        Normalize the frequencies.
    shift : bool
        Does shift the zero frequency to center.
    dtype : torch tensor type
        Data type, default is ``th.float32``.
    device : str
        device string, default is ``'cpu'``.

    Returns
    -------
    torch 1d-tensor
        Frequency array with size :math:`n×1`.

    Examples
    --------

    ::

        import torchbox as tb

        n = 10
        print(np.fft.fftfreq(n, d=0.1), 'numpy')
        print(th.fft.fftfreq(n, d=0.1), 'torch')
        print(tb.fftfreq(n, fs=10., norm=False), 'fftfreq, norm=False, shift=False')
        print(tb.fftfreq(n, fs=10., norm=True), 'fftfreq, norm=True, shift=False')
        print(tb.fftfreq(n, fs=10., shift=True), 'fftfreq, norm=False, shift=True')
        print(tb.freq(n, fs=10., norm=False), 'freq, norm=False, shift=False')
        print(tb.freq(n, fs=10., norm=True), 'freq, norm=True, shift=False')
        print(tb.freq(n, fs=10., shift=True), 'freq, norm=False, shift=True')

        # ---output
        [ 0.  1.  2.  3.  4. -5. -4. -3. -2. -1.] numpy
        tensor([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.]) torch
        tensor([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.]) fftfreq, norm=False, shift=False
        tensor([ 0.0000,  0.1000,  0.2000,  0.3000,  0.4000, -0.5000, -0.4000, -0.3000,
                -0.2000, -0.1000]) fftfreq, norm=True, shift=False
        tensor([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.]) fftfreq, norm=False, shift=True
        tensor([ 0.0000,  1.1111,  2.2222,  3.3333,  4.4444,  5.5556,  6.6667,  7.7778,
                8.8889, 10.0000]) freq, norm=False, shift=False
        tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889,
                1.0000]) freq, norm=True, shift=False
        tensor([-5.0000, -3.8889, -2.7778, -1.6667, -0.5556,  0.5556,  1.6667,  2.7778,
                3.8889,  5.0000]) freq, norm=False, shift=True
    """

def fftfreq(n, fs, norm=False, shift=False, dtype=th.float32, device='cpu'):
    r"""Return the Discrete Fourier Transform sample frequencies

    Return the Discrete Fourier Transform sample frequencies.

    Given a window length `n` and a sample rate `fs`, if shift is ``True``::

      f = [-n/2, ..., -1,     0, 1, ...,   n/2-1] / (d*n)   if n is even
      f = [-(n-1)/2, ..., -1, 0, 1, ..., (n-1)/2] / (d*n)   if n is odd

    Given a window length `n` and a sample rate `fs`, if shift is ``False``::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    If :attr:`norm` is ``True``, :math:`d = 1`, else :math:`d = 1/f_s`.

    Parameters
    ----------
    n : int
        Number of samples.
    fs : float
        Sampling rate.
    norm : bool
        Normalize the frequencies.
    shift : bool
        Does shift the zero frequency to center.
    dtype : torch tensor type
        Data type, default is ``th.float32``.
    device : torch device
        device string, ``'cpu', 'cuda:0', 'cuda:1'``

    Returns
    -------
    torch 1d-array
        Frequency array with size :math:`n×1`.

    Examples
    --------

    ::

        import torchbox as tb

        n = 10
        print(np.fft.fftfreq(n, d=0.1), 'numpy')
        print(th.fft.fftfreq(n, d=0.1), 'torch')
        print(tb.fftfreq(n, fs=10., norm=False), 'fftfreq, norm=False, shift=False')
        print(tb.fftfreq(n, fs=10., norm=True), 'fftfreq, norm=True, shift=False')
        print(tb.fftfreq(n, fs=10., shift=True), 'fftfreq, norm=False, shift=True')
        print(tb.freq(n, fs=10., norm=False), 'freq, norm=False, shift=False')
        print(tb.freq(n, fs=10., norm=True), 'freq, norm=True, shift=False')
        print(tb.freq(n, fs=10., shift=True), 'freq, norm=False, shift=True')

        # ---output
        [ 0.  1.  2.  3.  4. -5. -4. -3. -2. -1.] numpy
        tensor([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.]) torch
        tensor([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.]) fftfreq, norm=False, shift=False
        tensor([ 0.0000,  0.1000,  0.2000,  0.3000,  0.4000, -0.5000, -0.4000, -0.3000,
                -0.2000, -0.1000]) fftfreq, norm=True, shift=False
        tensor([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.]) fftfreq, norm=False, shift=True
        tensor([ 0.0000,  1.1111,  2.2222,  3.3333,  4.4444,  5.5556,  6.6667,  7.7778,
                8.8889, 10.0000]) freq, norm=False, shift=False
        tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889,
                1.0000]) freq, norm=True, shift=False
        tensor([-5.0000, -3.8889, -2.7778, -1.6667, -0.5556,  0.5556,  1.6667,  2.7778,
                3.8889,  5.0000]) freq, norm=False, shift=True

    """

def fftshift(x, **kwargs):
    r"""Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int, optional
        Axes over which to shift. (Default is None, which shifts all axes.)

    Returns
    -------
    y : Tensor
        The shifted tensor.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    ::

        import numpy as np
        import torch as th
        import torchbox as tb

        x = [1, 2, 3, 4, 5, 6]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x)
        print(y)

        x = [1, 2, 3, 4, 5, 6, 7]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x)
        print(y)

        dim = (0, 1)  # dim = 0, dim = 1
        x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
        y = np.fft.fftshift(x, dim)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x, dim)
        print(y)


        x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
        y = np.fft.fftshift(x, dim)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x, dim)
        print(y)

    """

def ifftshift(x, **kwargs):
    r"""Shift the zero-frequency component back.

    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    dim : int, optional
        Axes over which to shift. (Default is None, which shifts all axes.)

    Returns
    -------
    y : Tensor
        The shifted tensor.

    See Also
    --------
    fftshift : The inverse of `ifftshift`.

    Examples
    --------
    ::

        import numpy as np
        import torch as th
        import torchbox as tb

        x = [1, 2, 3, 4, 5, 6]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x)
        print(y)

        x = [1, 2, 3, 4, 5, 6, 7]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x)
        print(y)

        dim = (0, 1)  # dim = 0, dim = 1
        x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
        y = np.fft.fftshift(x, dim)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x, dim)
        print(y)


        x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
        y = np.fft.fftshift(x, dim)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x, dim)
        print(y)

    """

def padfft(X, nfft=None, dim=0, shift=False):
    r"""PADFT Pad array for doing FFT or IFFT

    PADFT Pad array for doing FFT or IFFT

    Parameters
    ----------
    X : Tensor
        Data to be padded.
    nfft : int or None
        Padding size.
    dim : int, optional
        Padding dimension. (the default is 0)
    shift : bool, optional
        Whether to shift the frequency (the default is False)

    Returns
    -------
    y : Tensor
        The padded tensor.
    """

def fft(x, n=None, norm="backward", shift=False, **kwargs):
    r"""FFT in torchbox

    FFT in torchbox, both real and complex valued tensors are supported.

    Parameters
    ----------
    x : Tensor
        When :attr:`x` is complex, it can be either in real-representation format or complex-representation format.
    n : int, optional
        The number of fft points (the default is None --> equals to signal dimension)
    norm : None or str, optional
        Normalization mode. For the forward transform (fft()), these correspond to:
        "forward" - normalize by ``1/n``; "backward" - no normalization (default); "ortho" - normalize by ``1/sqrt(n)`` (making the FFT orthonormal).
    shift : bool, optional
        shift the zero frequency to center (the default is False)
    cdim : int or None
        If :attr:`x` is complex-valued, :attr:`cdim` is ignored. If :attr:`x` is real-valued and :attr:`cdim` is integer
        then :attr:`x` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex dimension;
        otherwise (None), :attr:`x` will be treated as real-valued.
    dim : int, optional
        axis of fft operation (the default is 0, which means the first dimension)

    Returns
    -------
    y : Tensor
        fft results tensor with the same type as :attr:`x`

    Raises
    ------
    ValueError
        nfft is small than signal dimension.

    see also :func:`ifft`, :func:`fftfreq`, :func:`freq`.

    Examples
    ---------

    .. image:: ./_static/FFTIFFTdemo.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torch as th
        import torchbox as tb
        import matplotlib.pyplot as plt

        shift = True
        frq = [10, 10]
        amp = [0.8, 0.6]
        Fs = 80
        Ts = 2.
        Ns = int(Fs * Ts)

        t = th.linspace(-Ts / 2., Ts / 2., Ns).reshape(Ns, 1)
        f = tb.freq(Ns, Fs, shift=shift)
        f = tb.fftfreq(Ns, Fs, norm=False, shift=shift)

        # ---complex vector in real representation format
        x = amp[0] * th.cos(2. * th.pi * frq[0] * t) + 1j * amp[1] * th.sin(2. * th.pi * frq[1] * t)

        # ---do fft
        Xc = tb.fft(x, n=Ns, cdim=None, dim=0, shift=shift)

        # ~~~get real and imaginary part
        xreal = tb.real(x, cdim=None, keepdim=False)
        ximag = tb.imag(x, cdim=None, keepdim=False)
        Xreal = tb.real(Xc, cdim=None, keepdim=False)
        Ximag = tb.imag(Xc, cdim=None, keepdim=False)

        # ---do ifft
        x̂ = tb.ifft(Xc, n=Ns, cdim=None, dim=0, shift=shift)
        
        # ~~~get real and imaginary part
        x̂real = tb.real(x̂, cdim=None, keepdim=False)
        x̂imag = tb.imag(x̂, cdim=None, keepdim=False)

        plt.figure()
        plt.subplot(131)
        plt.grid()
        plt.plot(t, xreal)
        plt.plot(t, ximag)
        plt.legend(['real', 'imag'])
        plt.title('signal in time domain')
        plt.subplot(132)
        plt.grid()
        plt.plot(f, Xreal)
        plt.plot(f, Ximag)
        plt.legend(['real', 'imag'])
        plt.title('signal in frequency domain')
        plt.subplot(133)
        plt.grid()
        plt.plot(t, x̂real)
        plt.plot(t, x̂imag)
        plt.legend(['real', 'imag'])
        plt.title('reconstructed signal')
        plt.show()

        # ---complex vector in real representation format
        x = tb.c2r(x, cdim=-1)

        # ---do fft
        Xc = tb.fft(x, n=Ns, cdim=-1, dim=0, shift=shift)

        # ~~~get real and imaginary part
        xreal = tb.real(x, cdim=-1, keepdim=False)
        ximag = tb.imag(x, cdim=-1, keepdim=False)
        Xreal = tb.real(Xc, cdim=-1, keepdim=False)
        Ximag = tb.imag(Xc, cdim=-1, keepdim=False)

        # ---do ifft
        x̂ = tb.ifft(Xc, n=Ns, cdim=-1, dim=0, shift=shift)
        
        # ~~~get real and imaginary part
        x̂real = tb.real(x̂, cdim=-1, keepdim=False)
        x̂imag = tb.imag(x̂, cdim=-1, keepdim=False)

        plt.figure()
        plt.subplot(131)
        plt.grid()
        plt.plot(t, xreal)
        plt.plot(t, ximag)
        plt.legend(['real', 'imag'])
        plt.title('signal in time domain')
        plt.subplot(132)
        plt.grid()
        plt.plot(f, Xreal)
        plt.plot(f, Ximag)
        plt.legend(['real', 'imag'])
        plt.title('signal in frequency domain')
        plt.subplot(133)
        plt.grid()
        plt.plot(t, x̂real)
        plt.plot(t, x̂imag)
        plt.legend(['real', 'imag'])
        plt.title('reconstructed signal')
        plt.show()

    """

def ifft(x, n=None, norm="backward", shift=False, **kwargs):
    r"""IFFT in torchbox

    IFFT in torchbox, both real and complex valued tensors are supported.

    Parameters
    ----------
    x : Tensor
        When :attr:`x` is complex, it can be either in real-representation format or complex-representation format.
    n : int, optional
        The number of ifft points (the default is None --> equals to signal dimension)
    norm : bool, optional
        Normalization mode. For the backward transform (ifft()), these correspond to: "forward" - no normalization;
         "backward" - normalize by ``1/n`` (default); "ortho" - normalize by 1``/sqrt(n)`` (making the IFFT orthonormal).
    shift : bool, optional
        shift the zero frequency to center (the default is False)
    cdim : int or None
        If :attr:`x` is complex-valued, :attr:`cdim` is ignored. If :attr:`x` is real-valued and :attr:`cdim` is integer
        then :attr:`x` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`x` will be treated as real-valued.
    dim : int, optional
        axis of fft operation (the default is 0, which means the first dimension)

    Returns
    -------
    y : Tensor
        ifft results tensor with the same type as :attr:`x`

    Raises
    ------
    ValueError
        nfft is small than signal dimension.

    see also :func:`fft`, :func:`fftfreq`, :func:`freq`. see also :func:`fft` for examples. 

    """


