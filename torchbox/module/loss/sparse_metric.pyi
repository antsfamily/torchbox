class LogSparseLoss(th.nn.Module):
    """Log sparse loss

    Parameters
    ----------
    X : array
        the input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. The default is :obj:`None`, which means all. 
    lambd : float
        weight, default is 1.
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         loss
    """

    def __init__(self, lambd=1., cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, X):
        ...

class FourierLogSparseLoss(th.nn.Module):
    r"""FourierLogSparseLoss

    Parameters
    ----------
    X : array
        the input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. The default is :obj:`None`, which means all. 
    lambd : float
        weight, default is 1.
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is ``'mean'``)
    
    Returns
    -------
    scalar or array
         loss

    """

    def __init__(self, lambd=1., cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, X):
        ...


