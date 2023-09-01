class MSE(th.nn.Module):
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
        C1 = MSE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = MSE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MSE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = MSE(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = MSE(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MSE(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = MSE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = MSE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MSE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

class SSE(th.nn.Module):
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
        C1 = SSE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = SSE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SSE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = SSE(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = SSE(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SSE(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = SSE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = SSE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SSE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

class MAE(th.nn.Module):
    r"""computes the mean absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm MAE}({\bf P, G}) = \frac{1}{N}\||{\bf P} - {\bf G}|\| = \frac{1}{N}\sum_{i=1}^N |p_i - g_i|

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
         mean absoluted error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = MAE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = MAE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MAE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = MAE(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = MAE(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MAE(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = MAE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = MAE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = MAE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

class SAE(th.nn.Module):
    r"""computes the sum absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm SAE}({\bf P, G}) = \||{\bf P} - {\bf G}|\| = \sum_{i=1}^N |p_i - g_i|

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
         sum absoluted error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = SAE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = SAE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SAE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = SAE(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = SAE(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SAE(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = SAE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = SAE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = SAE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

class NMSE(th.nn.Module):
    r"""computes the normalized mean square error

    Both complex and real representation are supported.

    .. math::
       {\rm MSE}({\bf P, G}) = \frac{\frac{1}{N}\|{\bf P} - {\bf G}\|_2^2}{\|{\bf G}\|_2^2} = \frac{\frac{1}{N}\sum_{i=1}^N(|p_i - g_i|)^2}{\sum_{i=1}^N(|p_i - g_i|)^2}

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
        C1 = NMSE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMSE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMSE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = NMSE(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMSE(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMSE(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = NMSE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMSE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMSE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

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

class NSSE(th.nn.Module):
    r"""computes the normalized sum square error

    Both complex and real representation are supported.

    .. math::
       {\rm SSE}({\bf P, G}) = \frac{\|{\bf P} - {\bf G}\|_2^2}{\|{\bf G}\|_2^2} = \frac{\sum_{i=1}^N(|p_i - g_i|)^2}{\sum_{i=1}^N(|g_i|)^2}

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
        C1 = NSSE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSSE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSSE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = NSSE(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSSE(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSSE(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = NSSE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSSE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSSE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

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

class NMAE(th.nn.Module):
    r"""computes the normalized mean absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm MAE}({\bf P, G}) = \frac{\frac{1}{N}\||{\bf P} - {\bf G}|\|}{\||{\bf G}|\|} = \frac{\frac{1}{N}\sum_{i=1}^N |p_i - g_i|}{\sum_{i=1}^N |p_i - g_i|}

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
         mean absoluted error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = NMAE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMAE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMAE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = NMAE(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMAE(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMAE(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = NMAE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NMAE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NMAE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

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

class NSAE(th.nn.Module):
    r"""computes the normalized sum absoluted error

    Both complex and real representation are supported.

    .. math::
       {\rm SAE}({\bf P, G}) = \frac{\||{\bf P} - {\bf G}|\|}{\|{\bf G}|\|} = \frac{\sum_{i=1}^N |p_i - g_i|}{\sum_{i=1}^N |g_i|}

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
         sum absoluted error

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        C1 = NSAE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSAE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSAE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in real format
        C1 = NSAE(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSAE(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSAE(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = NSAE(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        C2 = NSAE(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        C3 = NSAE(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(C1, C2, C3)

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


