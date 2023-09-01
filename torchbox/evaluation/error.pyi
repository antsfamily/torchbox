def mse(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the mean square error

    Both complex and real representation are supported.

    .. math::
       {\rm MSE}({\bf P, G}) = \frac{1}{N}\|{\bf P} - {\bf G}\|_2^2 = \frac{1}{N}\sum_{i=1}^N(|p_i - g_i|)^2

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
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

    """

def sse(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the sum square error

    Both complex and real representation are supported.

    .. math::
       {\rm SSE}({\bf P, G}) = \|{\bf P} - {\bf G}\|_2^2 = \sum_{i=1}^N(|p_i - g_i|)^2

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
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

    """

def mae(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the mean absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm MAE}({\bf P, G}) = \frac{1}{N}|{\bf P} - {\bf G}| = \frac{1}{N}\sum_{i=1}^N |p_i - g_i|

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
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

    """

def sae(P, G, cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the sum absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm SAE}({\bf P, G}) = |{\bf P} - {\bf G}| = \sum_{i=1}^N |p_i - g_i|

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    cdim : int or None
        If :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing error. 
        The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
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

    """

def nmse(P, G, mode='Gpowsum', cdim=None, dim=None, keepdim=False, reduction='mean'):
    r"""computes the normalized mean square error

    Both complex and real representation are supported.

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
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
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
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
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
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
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
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
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
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
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
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
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
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
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
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


