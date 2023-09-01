def unwrap(x, discont=tb.PI, axis=-1, imp=None):
    r"""Unwrap by changing deltas between values to :math:`2\pi` complement.

    Unwrap radian phase `x` by changing absolute jumps greater than
    `discont` to their :math:`2\pi` complement along the given axis.

    Parameters
    ----------
    x : Tensor or ndarray
        The input.
    discont : float, optional
        Maximum discontinuity between values, default is :math:`\pi`.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.
    imp : str, optional
        The implenmentation way, ``'numpy'`` --> numpy, None or ``'torch'`` --> torch (default)

    Returns
    -------
    Tensor or ndarray
        The unwrapped.
    
    Examples
    --------

    ::

        x_np = np.array([3.14, -3.12, 3.12, 3.13, -3.11])
        y_np = unwrap(x_np)
        print(y_np, y_np.shape, type(y_np))
        x_th = th.Tensor(x_np)
        y_th = unwrap(x_th)
        print(y_th, y_th.shape, type(y_th))

        print("------------------------")
        x_th = th.tensor([3.14, -3.12, 3.12, 3.13, -3.11])
        x_th = th.cat((th.flip(x_th, dims=[0]), x_th), axis=0)
        print(x_th)
        y_th = unwrap2(x_th)
        print(y_th, y_th.shape, type(y_th))

        print("------------------------")
        x_np = np.array([3.14, -3.12, 3.12, 3.13, -3.11])
        x_th = th.Tensor(x_np)
        y_th = unwrap(x_th, imp='numpy')
        print(y_th, y_th.shape, type(y_th))

        y_th = unwrap(x_th, imp=None)
        print(y_th, y_th.shape, type(y_th))

        # output

        tensor([3.1400, 3.1632, 3.1200, 3.1300, 3.1732], dtype=torch.float64) torch.Size([5]) <class 'torch.Tensor'>
        tensor([3.1400, 3.1632, 3.1200, 3.1300, 3.1732]) torch.Size([5]) <class 'torch.Tensor'>
        ------------------------
        tensor([-3.1100,  3.1300,  3.1200, -3.1200,  3.1400,  3.1400, -3.1200,  3.1200,
                3.1300, -3.1100])
        tensor([3.1732, 3.1300, 3.1200, 3.1632, 3.1400, 3.1400, 3.1632, 3.1200, 3.1300,
                3.1732]) torch.Size([10]) <class 'torch.Tensor'>
        ------------------------
        tensor([3.1400, 3.1632, 3.1200, 3.1300, 3.1732]) torch.Size([5]) <class 'torch.Tensor'>
        tensor([3.1400, 3.1632, 3.1200, 3.1300, 3.1732]) torch.Size([5]) <class 'torch.Tensor'>

    """

def unwrap2(x, discont=tb.PI, axis=-1):
    r"""Unwrap by changing deltas between values to :math:`2\pi` complement.

    Unwrap radian phase `x` by changing absolute jumps greater than
    `discont` to their :math:`2\pi` complement along the given axis. The elements
    are divided into 2 parts (with equal length) along the given axis.
    The first part is unwrapped in inverse order, while the second part
    is unwrapped in normal order.

    Parameters
    ----------
    x : Tensor
        The input.
    discont : float, optional
        Maximum discontinuity between values, default is :math:`\pi`.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.

    Returns
    -------
    Tensor
        The unwrapped.

    see also :func:`unwrap`
    """


