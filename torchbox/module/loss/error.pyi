class MSELoss(th.nn.Module):
    r"""computes the mean square error

    Both complex and real representation are supported.

    .. math::
       {\rm MSE}({\bf P, G}) = \frac{1}{N}\|{\bf P} - {\bf G}\|_2^2 = \frac{1}{N}\sum_{i=1}^N(|p_i - g_i|)^2

    Parameters
    ----------
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
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
        C1 = MSELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = MSELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MSELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = MSELoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = MSELoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MSELoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = MSELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = MSELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MSELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

    def __init__(self, cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """     

class SSELoss(th.nn.Module):
    r"""computes the sum square error

    Both complex and real representation are supported.

    .. math::
       {\rm SSE}({\bf P, G}) = \|{\bf P} - {\bf G}\|_2^2 = \sum_{i=1}^N(|p_i - g_i|)^2

    Parameters
    ----------
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
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
        C1 = SSELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = SSELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SSELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = SSELoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = SSELoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SSELoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = SSELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = SSELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SSELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

    def __init__(self, cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """

class MAELoss(th.nn.Module):
    r"""computes the mean absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm MAE}({\bf P, G}) = \frac{1}{N}|{\bf P} - {\bf G}| = \frac{1}{N}\sum_{i=1}^N |p_i - g_i|

    Parameters
    ----------
    P : array
        original
    P : array
        reconstructed
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
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
        C1 = MAELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = MAELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MAELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = MAELoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = MAELoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MAELoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = MAELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = MAELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MAELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

    def __init__(self, cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """

class SAELoss(th.nn.Module):
    r"""computes the sum absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm SAE}({\bf P, G}) = |{\bf P} - {\bf G}| = \sum_{i=1}^N |p_i - g_i|

    Parameters
    ----------
    P : array
        original
    P : array
        reconstructed
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
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
        C1 = SAELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = SAELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SAELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = SAELoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = SAELoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SAELoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = SAELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = SAELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SAELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

    def __init__(self, cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """     

class NMSELoss(th.nn.Module):
    r"""computes the normalized mean square error

    Both complex and real representation are supported.

    Parameters
    ----------
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

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = NMSELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMSELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMSELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = NMSELoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMSELoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMSELoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = NMSELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMSELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMSELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

    """

    def __init__(self, mode='Gpowsum', cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """

class NSSELoss(th.nn.Module):
    r"""computes the normalized sum square error

    Both complex and real representation are supported.

    Parameters
    ----------
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
         sum square error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = NSSELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSSELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSSELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = NSSELoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSSELoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSSELoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = NSSELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSSELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSSELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

    """

    def __init__(self, mode='Gpowsum', cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """

class NMAELoss(th.nn.Module):
    r"""computes the normalized mean absoluted error

    Both complex and real representation are supported.

    Parameters
    ----------
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

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = NMAELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMAELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMAELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = NMAELoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMAELoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMAELoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = NMAELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMAELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMAELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

    """

    def __init__(self, mode='Gabssum', cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """

class NSAELoss(th.nn.Module):
    r"""computes the normalized sum absoluted error

    Both complex and real representation are supported.

    Parameters
    ----------
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

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = NSAELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSAELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSAELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = NSAELoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSAELoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSAELoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = NSAELoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSAELoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSAELoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

    """

    def __init__(self, mode='Gabssum', cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        """


