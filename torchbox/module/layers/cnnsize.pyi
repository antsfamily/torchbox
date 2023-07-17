def conv_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
    r"""computes output shape of convolution

    .. math::
       \begin{array}{l}
       H_{o} &= \left\lfloor\frac{H_{i}  + 2 \times P_h - D_h \times (K_h - 1) - 1}{S_h} + 1\right\rfloor \\
       W_{o} &= \left\lfloor\frac{W_{i}  + 2 \times P_w - D_w \times (K_w - 1) - 1}{S_w} + 1\right\rfloor \\
       B_{o} &= \left\lfloor\frac{B_{i}  + 2 \times P_b - D_w \times (K_b - 1) - 1}{S_b} + 1\right\rfloor \\
        \cdots
       \end{array}
       :label: equ-DilationConvxdSize

    Parameters
    ----------
    in_size : list or tuple
        the size of input (without batch and channel)
    kernel_size : int, list or tuple
        the window size of convolution
    stride : int, list or tuple, optional
        the stride of convolution, by default 1
    padding : int, str, list or tuple, optional
        the padding size of convolution, ``'valid'``, ``'same'``, by default 0
    dilation : int, list or tuple, optional
        the spacing between kernel elements, by default 1
    """

def ConvSize1d(CLi, Co, K, S, P, D=1, groups=1):
    r"""Compute shape after 2D-Convolution

    .. math::
       \begin{array}{l}
       L_{o} &= \left\lfloor\frac{L_{i}  + 2 \times P_l - D_l \times (K_l - 1) - 1}{S_l} + 1\right\rfloor \\
       \end{array}

    CLi : tuple or list
        input data shape (C, L)
    Co : int
        number of output chanels.
    K : tuple
        kernel size
    S : tuple
        stride size
    P : tuple
        padding size
    D : tuple, optional
        dilation size (the default is 1)
    groups : int, optional
        1 (the default is 1)

    Returns
    -------
    tuple
        shape after 2D-Convolution

    Raises
    ------
    ValueError
        dilation should be greater than zero.
    """

def ConvTransposeSize1d(CLi, Co, K, S, P, D=1, OP=0, groups=1):
    r"""Compute shape after Transpose Convolution

    .. math::
       \begin{array}{l}
       L_{o} &= (L_{i} - 1) \times S_l - 2 \times P_l + D_l \times (K_l - 1) + OP_l + 1 \\
       \end{array}
       :label: equ-TransposeConv1dSize

    Parameters
    ----------
    CLi : tuple or list
        input data shape (C, H, W)
    Co : int
        number of output chanels.
    K : tuple
        kernel size
    S : tuple
        stride size
    P : tuple
        padding size
    D : tuple, optional
        dilation size (the default is 1)
    OP : tuple, optional
        output padding size (the default is 0)
    groups : int, optional
        one group (the default is 1)

    Returns
    -------
    tuple
        shape after 2D-Transpose Convolution

    Raises
    ------
    ValueError
        output padding must be smaller than either stride or dilation
    """

def PoolSize1d(CLi, K, S, P, D=1):
    ...

def UnPoolSize1d(CLi, K, S, P, D=1):
    ...

def ConvSize2d(CHWi, Co, K, S, P, D=(1, 1), groups=1):
    r"""Compute shape after 2D-Convolution

    .. math::
       \begin{array}{l}
       H_{o} &= \left\lfloor\frac{H_{i}  + 2 \times P_h - D_h \times (K_h - 1) - 1}{S_h} + 1\right\rfloor \\
       W_{o} &= \left\lfloor\frac{W_{i}  + 2 \times P_w - D_w \times (K_w - 1) - 1}{S_w} + 1\right\rfloor
       \end{array}
       :label: equ-DilationConv2dSize

    CHWi : tuple or list
        input data shape (C, H, W)
    Co : int
        number of output chanels.
    K : tuple
        kernel size
    S : tuple
        stride size
    P : tuple
        padding size
    D : tuple, optional
        dilation size (the default is (1, 1))
    groups : int, optional
        [description] (the default is 1, which [default_description])

    Returns
    -------
    tuple
        shape after 2D-Convolution

    Raises
    ------
    ValueError
        dilation should be greater than zero.
    """

def ConvTransposeSize2d(CHWi, Co, K, S, P, D=(1, 1), OP=(0, 0), groups=1):
    r"""Compute shape after Transpose Convolution

    .. math::
       \begin{array}{l}
       H_{o} &= (H_{i} - 1) \times S_h - 2 \times P_h + D_h \times (K_h - 1) + OP_h + 1 \\
       W_{o} &= (W_{i} - 1) \times S_w - 2 \times P_w + D_w \times (K_w - 1) + OP_w + 1
       \end{array}
       :label: equ-TransposeConv2dSize

    Parameters
    ----------
    CHWi : tuple or list
        input data shape (C, H, W)
    Co : int
        number of output chanels.
    K : tuple
        kernel size
    S : tuple
        stride size
    P : tuple
        padding size
    D : tuple, optional
        dilation size (the default is (1, 1))
    OP : tuple, optional
        output padding size (the default is (0, 0))
    groups : int, optional
        one group (the default is 1)

    Returns
    -------
    tuple
        shape after 2D-Transpose Convolution

    Raises
    ------
    ValueError
        output padding must be smaller than either stride or dilation
    """

def PoolSize2d(CHWi, K, S, P, D=(1, 1)):
    ...

def UnPoolSize2d(CHWi, K, S, P, D=(1, 1)):
    ...


