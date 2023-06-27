def mse(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the mean square error

    Both complex and real representation are supported.

    .. math::
       {\rm MSE}({\bf P, G}) = \frac{1}{N}\|{\bf P} - {\bf G}\|_2^2 = \frac{1}{N}\sum_{i=1}^N(|x_i - y_i|)^2

    Parameters
    ----------
    P : array
        reconstructed
    G : array
        target or ground-truth
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
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
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = mse(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = mse(P, G, cdim=1, dim=(-2, -1), reduction=None)
        C2 = mse(P, G, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = mse(P, G, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = mse(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
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

def sse(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the sum square error

    Both complex and real representation are supported.

    .. math::
       {\rm SSE}({\bf P, G}) = \|{\bf P} - {\bf G}\|_2^2 = \sum_{i=1}^N(|x_i - y_i|)^2

    Parameters
    ----------
    P : array
        reconstructed
    G : array
        target or ground-truth
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
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
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = sse(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = sse(P, G, cdim=1, dim=(-2, -1), reduction=None)
        C2 = sse(P, G, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = sse(P, G, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = sse(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sse(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sse(P, G, cdim=None, dim=(-2, -1), reduction='mean')
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

def mae(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the mean absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm MAE}({\bf P, G}) = \frac{1}{N}|{\bf P} - {\bf G}| = \frac{1}{N}\sum_{i=1}^N |x_i - y_i|

    Parameters
    ----------
    P : array
        reconstructed
    G : array
        target or ground-truth
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
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
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = mae(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = mae(P, G, cdim=1, dim=(-2, -1), reduction=None)
        C2 = mae(P, G, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = mae(P, G, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = mae(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = mae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = mae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
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

def sae(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the sum absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm SAE}({\bf P, G}) = |{\bf P} - {\bf G}| = \sum_{i=1}^N |x_i - y_i|

    Parameters
    ----------
    P : array
        reconstructed
    G : array
        target or ground-truth
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
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
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = sae(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = sae(P, G, cdim=1, dim=(-2, -1), reduction=None)
        C2 = sae(P, G, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = sae(P, G, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = sae(P, G, cdim=None, dim=(-2, -1), reduction=None)
        C2 = sae(P, G, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = sae(P, G, cdim=None, dim=(-2, -1), reduction='mean')
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

def nmse(P, G, mode='Gpowsum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized mean square error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : array
        reconstructed
    G : array
        target or ground-truth
    mode : str
        mode of normalization
        ``'Gpowsum'`` (default) normalized square error with the power summation of :attr:`G`, 
        ``'Gabssum'`` (default) normalized square error with the amplitude summation of :attr:`G`, 
        ``'Gpowmax'`` normalized square error with the maximum power of :attr:`G`,
        ``'Gabsmax'`` normalized square error with the maximum amplitude of :attr:`G`,
        ``'GpeakV'`` normalized square error with the square of peak value (V) of :attr:`G`;
        ``'Gfnorm'`` normalized square error with Frobenius norm of :attr:`G`;
        ``'Gpnorm'`` normalized square error with p-norm of :attr:`G`;
        ``'fnorm'`` normalized :attr:`P` and :attr:`G` with Frobenius norm,
        ``'pnormV'`` normalized :attr:`P` and :attr:`G` with p-norm, respectively, where V is a float or integer number; 
        ``'zscore'`` normalized :attr:`P` and :attr:`G` with zscore method.
        ``'std'`` normalized :attr:`P` and :attr:`G` with standard deviation.
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
        normalized mean square error

    Examples
    ---------

    ::

        mode = 'Gabssum'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nmse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

def nsse(P, G, mode='Gpowsum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized sum square error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : array
        reconstructed
    G : array
        target or ground-truth
    mode : str
        mode of normalization, 
        ``'Gpowsum'`` (default) normalized square error with the power summation of :attr:`G`, 
        ``'Gabssum'`` (default) normalized square error with the amplitude summation of :attr:`G`, 
        ``'Gpowmax'`` normalized square error with the maximum power of :attr:`G`,
        ``'Gabsmax'`` normalized square error with the maximum amplitude of :attr:`G`,
        ``'GpeakV'`` normalized square error with the square of peak value (V) of :attr:`G`;
        ``'Gfnorm'`` normalized square error with Frobenius norm of :attr:`G`;
        ``'Gpnorm'`` normalized square error with p-norm of :attr:`G`;
        ``'fnorm'`` normalized :attr:`P` and :attr:`G` with Frobenius norm,
        ``'pnormV'`` normalized :attr:`P` and :attr:`G` with p-norm, respectively, where V is a float or integer number; 
        ``'zscore'`` normalized :attr:`P` and :attr:`G` with zscore method.
        ``'std'`` normalized :attr:`P` and :attr:`G` with standard deviation.
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
        normalized sum square error

    Examples
    ---------

    ::

        mode = 'Gabssum'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nsse(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsse(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """

def nmae(P, G, mode='Gabssum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized mean absoluted error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : array
        reconstructed
    G : array
        target or ground-truth
    mode : str
        mode of normalization, 
        ``'Gabssum'`` (default) normalized square error with the amplitude summation of :attr:`G`, 
        ``'Gpowsum'`` normalized square error with the power summation of :attr:`G`, 
        ``'Gabsmax'`` normalized square error with the maximum amplitude of :attr:`G`,
        ``'Gpowmax'`` normalized square error with the maximum power of :attr:`G`,
        ``'GpeakV'`` normalized square error with the square of peak value (V) of :attr:`G`;
        ``'Gfnorm'`` normalized square error with Frobenius norm of :attr:`G`;
        ``'Gpnorm'`` normalized square error with p-norm of :attr:`G`;
        ``'fnorm'`` normalized :attr:`P` and :attr:`G` with Frobenius norm,
        ``'pnormV'`` normalized :attr:`P` and :attr:`G` with p-norm, respectively, where V is a float or integer number; 
        ``'zscore'`` normalized :attr:`P` and :attr:`G` with zscore method.
        ``'std'`` normalized :attr:`P` and :attr:`G` with standard deviation.
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         normalized mean absoluted error

    Examples
    ---------

    ::

        mode = 'Gabssum'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nmae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nmae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)
  
    """

def nsae(P, G, mode='Gabssum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized sum absoluted error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : array
        reconstructed
    G : array
        target or ground-truth
    mode : str
        mode of normalization, 
        ``'Gabssum'`` (default) normalized square error with the amplitude summation of :attr:`G`, 
        ``'Gpowsum'`` normalized square error with the power summation of :attr:`G`, 
        ``'Gabsmax'`` normalized square error with the maximum amplitude of :attr:`G`,
        ``'Gpowmax'`` normalized square error with the maximum power of :attr:`G`,
        ``'GpeakV'`` normalized square error with the square of peak value (V) of :attr:`G`;
        ``'Gfnorm'`` normalized square error with Frobenius norm of :attr:`G`;
        ``'Gpnorm'`` normalized square error with p-norm of :attr:`G`;
        ``'fnorm'`` normalized :attr:`P` and :attr:`G` with Frobenius norm,
        ``'pnormV'`` normalized :attr:`P` and :attr:`G` with p-norm, respectively, where V is a float or integer number; 
        ``'zscore'`` normalized :attr:`P` and :attr:`G` with zscore method.
        ``'std'`` normalized :attr:`P` and :attr:`G` with standard deviation.
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         sum absoluted error

    Examples
    ---------

    ::

        mode = 'Gabssum'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction=None)
        C2 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='sum')
        C3 = nsae(P, G, mode=mode, cdim=1, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction=None)
        C2 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='sum')
        C3 = nsae(P, G, mode=mode, cdim=None, dim=(-2, -1), reduction='mean')
        print(C1, C2, C3)

    """


