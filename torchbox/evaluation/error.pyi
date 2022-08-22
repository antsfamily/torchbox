def mse(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""computes the mean square error

    Both complex and real representation are supported.

    .. math::
       {\rm MSE}({\bf X, Y}) = \frac{1}{N}\|{\bf X} - {\bf Y}\|_2^2 = \frac{1}{N}\sum_{i=1}^N(|x_i - y_i|)^2

    Parameters
    ----------
    X : array
        reconstructed
    Y : array
        target
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing error. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         mean square error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)

        # real
        C1 = mse(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mse(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mse(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = mse(X, Y, cdim=1, dim=(-2, -1), reduction=None)
        C2 = mse(X, Y, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = mse(X, Y, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = mse(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mse(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mse(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # ---output
        [[1.57602573 2.32844311]
        [1.07232374 2.36118382]
        [2.1841515  0.79002805]
        [2.43036295 3.18413899]
        [2.31107373 2.73990485]] 20.977636476183186 2.0977636476183186
        [3.90446884 3.43350757 2.97417955 5.61450194 5.05097858] 20.977636476183186 4.195527295236637
        [3.90446884 3.43350757 2.97417955 5.61450194 5.05097858] 20.977636476183186 4.195527295236637

    """

def sse(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""computes the sum square error

    Both complex and real representation are supported.

    .. math::
       {\rm SSE}({\bf X, Y}) = \|{\bf X} - {\bf Y}\|_2^2 = \sum_{i=1}^N(|x_i - y_i|)^2

    Parameters
    ----------
    X : array
        reconstructed
    Y : array
        target
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing error. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         sum square error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)

        # real
        C1 = sse(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sse(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sse(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = sse(X, Y, cdim=1, dim=(-2, -1), reduction=None)
        C2 = sse(X, Y, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = sse(X, Y, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = sse(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sse(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sse(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # ---output
        [[18.91230872 27.94131733]
        [12.86788492 28.33420589]
        [26.209818    9.48033663]
        [29.16435541 38.20966786]
        [27.73288477 32.87885818]] 251.73163771419823 25.173163771419823
        [46.85362605 41.20209081 35.69015462 67.37402327 60.61174295] 251.73163771419823 50.346327542839646
        [46.85362605 41.20209081 35.69015462 67.37402327 60.61174295] 251.73163771419823 50.346327542839646

    """

def mae(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""computes the mean absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm MAE}({\bf X, Y}) = \frac{1}{N}\||{\bf X} - {\bf Y}|\| = \frac{1}{N}\sum_{i=1}^N |x_i - y_i|

    Parameters
    ----------
    X : array
        original
    X : array
        reconstructed
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing error. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         mean absoluted error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)

        # real
        C1 = mae(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mae(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mae(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = mae(X, Y, cdim=1, dim=(-2, -1), reduction=None)
        C2 = mae(X, Y, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = mae(X, Y, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = mae(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mae(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mae(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # ---output
        [[1.06029116 1.19884877]
        [0.90117091 1.13552361]
        [1.23422083 0.75743914]
        [1.16127965 1.42169262]
        [1.25090731 1.29134222]] 11.41271620974502 1.141271620974502
        [1.71298566 1.50327364 1.53328572 2.11430946 2.01435599] 8.878210471231741 1.7756420942463482
        [1.71298566 1.50327364 1.53328572 2.11430946 2.01435599] 8.878210471231741 1.7756420942463482

    """

def sae(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""computes the sum absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm SAE}({\bf X, Y}) = \||{\bf X} - {\bf Y}|\| = \sum_{i=1}^N |x_i - y_i|

    Parameters
    ----------
    X : array
        original
    X : array
        reconstructed
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing error. 
        The default is :obj:`None`, which means all.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
        sum absoluted error

    Examples
    ---------

    ::
        norm = False
        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)

        # real
        C1 = sae(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sae(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sae(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = sae(X, Y, cdim=1, dim=(-2, -1), reduction=None)
        C2 = sae(X, Y, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = sae(X, Y, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = sae(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sae(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sae(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # ---output
        [[12.72349388 14.3861852 ]
        [10.81405096 13.62628335]
        [14.81065     9.08926963]
        [13.93535577 17.0603114 ]
        [15.0108877  15.49610662]] 136.95259451694022 13.695259451694023
        [20.55582795 18.03928365 18.39942858 25.37171356 24.17227192] 106.53852565478087 21.307705130956172
        [20.55582795 18.03928365 18.39942858 25.37171356 24.17227192] 106.5385256547809 21.30770513095618

    """

def nmse(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""computes the normalized mean square error

    Both complex and real representation are supported.

    .. math::
       {\rm NMSE}({\bf X, Y}) = \frac{\frac{1}{N}\|{\bf X} - {\bf Y}\|_2^2}{\|{\bf Y}\|_2^2}= \frac{\frac{1}{N}\sum_{i=1}^N(|x_i - y_i|)^2}{\sum_{i=1}^N(|y_i|)^2}

    Parameters
    ----------
    X : array
        reconstructed
    Y : array
        target
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing error. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
        normalized mean square error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)

        # real
        C1 = nmse(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmse(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmse(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nmse(X, Y, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nmse(X, Y, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nmse(X, Y, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = nmse(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmse(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmse(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

def nsse(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""computes the normalized sum square error

    Both complex and real representation are supported.

    .. math::
       {\rm NSSE}({\bf X, Y}) = \frac{\|{\bf X} - {\bf Y}\|_2^2}{\|{\bf Y}\|_2^2} = \frac{\sum_{i=1}^N(|x_i - y_i|)^2}{\sum_{i=1}^N(|y_i|)^2}

    Parameters
    ----------
    X : array
        reconstructed
    Y : array
        target
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing error. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
        normalized sum square error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)

        # real
        C1 = nsse(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsse(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsse(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nsse(X, Y, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nsse(X, Y, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nsse(X, Y, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = nsse(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsse(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsse(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

def nmae(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""computes the normalized mean absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm NMAE}({\bf X, Y}) = \frac{\frac{1}{N}\||{\bf X} - {\bf Y}|\|}{\||{\bf Y}|\|}

    Parameters
    ----------
    X : array
        original
    X : array
        reconstructed
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing error. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         normalized mean absoluted error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)

        # real
        C1 = nmae(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmae(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmae(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nmae(X, Y, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nmae(X, Y, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nmae(X, Y, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = nmae(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmae(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmae(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)
  
    """

def nsae(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):
    r"""computes the normalized sum absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm NSAE}({\bf X, Y}) = \frac{\||{\bf X} - {\bf Y}|\|}{\||{\bf Y}|\|} = \frac{\sum_{i=1}^N |x_i - y_i|}{\sum_{i=1}^N |y_i|}

    Parameters
    ----------
    X : array
        original
    X : array
        reconstructed
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing error. 
        The default is :obj:`None`, which means all.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         sum absoluted error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)

        # real
        C1 = nsae(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsae(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsae(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nsae(X, Y, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nsae(X, Y, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nsae(X, Y, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = nsae(X, Y, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsae(X, Y, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsae(X, Y, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """


