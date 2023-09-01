def window(n, wtype=None, isperiodic=True, dtype=None, device=None, requires_grad=False):
    r"""Generates window

    Parameters
    ----------
    n : int
        The length of the window.
    wtype : str or None, optional
        The window type:
        - ``'rectangle'`` for rectangle window
        - ``'bartlett'`` for bartlett window
        - ``'blackman'`` for blackman window
        - ``'hamming x y'`` for hamming window with :math:`\alpha=x, \beta=y`, default is 0.54, 0.46.
        - ``'hanning'`` for hanning window
        - ``'kaiser x'`` for kaiser window with :math:`\beta=x`, default is 12.
    isperiodic : bool, optional
        If True (default), returns a window to be used as periodic function.
        If False, return a symmetric window.
    dtype : None, optional
        The desired data type of returned tensor.
    device : None, optional
        The desired device of returned tensor.
    requires_grad : bool, optional
        If autograd should record operations on the returned tensor. Default: False.

    Returns
    -------
    tensor
        A 1-D tensor of size (n,) containing the window
    """

def windowing(x, w, axis=None):
    """Performs windowing operation in the specified axis.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    w : Tensor
        A 1-d window tensor.
    axis : int or None, optional
        The axis.

    Returns
    -------
    tensor
        The windowed data.

    """


