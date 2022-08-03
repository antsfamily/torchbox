class ReciprocalContrastLoss(th.nn.Module):
    r"""ReciprocalContrastLoss

    way1 is defined as follows, for contrast, see [1]:

    .. math::
       C = \frac{{\rm E}(|I|^2)}{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}


    way2 is defined as follows, for contrast, see [2]:

    .. math::
       C = \frac{\left({\rm E}(|I|)\right)^2}{{\rm E}(|I|^2)}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"

    Parameters
    ----------
    X : torch tensor
        The image array.
    mode : str, optional
        ``'way1'`` or ``'way2'``
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing contrast. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    C : scalar or tensor
        The contrast value of input.

    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)

        # real
        C1 = ReciprocalContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
        C2 = ReciprocalContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
        C3 = ReciprocalContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        # complex in real format
        C1 = ReciprocalContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction=None)(X)
        C2 = ReciprocalContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction='sum')(X)
        C3 = ReciprocalContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        C1 = ReciprocalContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
        C2 = ReciprocalContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
        C3 = ReciprocalContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        tensor([[0.7929, 0.9021],
                [0.6253, 0.8248],
                [1.2193, 1.0114],
                [0.6956, 0.9909],
                [0.8774, 0.8432]]) tensor(8.7830) tensor(0.8783)
        tensor([1.5821, 0.8469, 1.6997, 0.8813, 1.6563]) tensor(6.6663) tensor(1.3333)
        tensor([1.5821, 0.8469, 1.6997, 0.8813, 1.6563]) tensor(6.6663) tensor(1.3333)
    """

    def __init__(self, mode='way1', cdim=None, dim=None, keepcdim=False, reduction='mean'):
        ...

    def forward(self, X):
        ...

class NegativeContrastLoss(th.nn.Module):
    r"""Negative Contrast Loss

    way1 is defined as follows, see [1]:

    .. math::
       C = -\frac{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}{{\rm E}(|I|^2)}


    way2 is defined as follows, see [2]:

    .. math::
       C = -\frac{{\rm E}(|I|^2)}{\left({\rm E}(|I|)\right)^2}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"

    Parameters
    ----------
    X : torch tensor
        The image tensor.
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing contrast. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    mode : str, optional
        ``'way1'`` or ``'way2'``
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    C : scalar or tensor
        The contrast value of input.
    
    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)

        # real
        C1 = NegativeContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
        C2 = NegativeContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
        C3 = NegativeContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        # complex in real format
        C1 = NegativeContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction=None)(X)
        C2 = NegativeContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction='sum')(X)
        C3 = NegativeContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        C1 = NegativeContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
        C2 = NegativeContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
        C3 = NegativeContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)


        # output
        tensor([[-1.2612, -1.1085],
                [-1.5992, -1.2124],
                [-0.8201, -0.9887],
                [-1.4376, -1.0091],
                [-1.1397, -1.1860]]) tensor(-11.7626) tensor(-1.1763)
        tensor([-0.6321, -1.1808, -0.5884, -1.1346, -0.6038]) tensor(-4.1396) tensor(-0.8279)
        tensor([-0.6321, -1.1808, -0.5884, -1.1346, -0.6038]) tensor(-4.1396) tensor(-0.8279)

    """

    def __init__(self, mode='way1', cdim=None, dim=None, keepcdim=False, reduction='mean'):
        ...

    def forward(self, X):
        ...

class ContrastLoss(th.nn.Module):
    r"""Contrast

    way1 is defined as follows, see [1]:

    .. math::
       C = \frac{\sqrt{{\rm E}\left(|I|^2 - {\rm E}(|I|^2)\right)^2}}{{\rm E}(|I|^2)}


    way2 is defined as follows, see [2]:

    .. math::
       C = \frac{{\rm E}(|I|^2)}{\left({\rm E}(|I|)\right)^2}

    [1] Efficient Nonparametric ISAR Autofocus Algorithm Based on Contrast Maximization and Newton
    [2] section 13.4.1 in "Ian G. Cumming's SAR book"

    Parameters
    ----------
    X : torch tensor
        The image tensor.
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing contrast. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    mode : str, optional
        ``'way1'`` or ``'way2'``
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    C : scalar or tensor
        The contrast value of input.
    
    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)

        # real
        C1 = ContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
        C2 = ContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
        C3 = ContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        # complex in real format
        C1 = ContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction=None)(X)
        C2 = ContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction='sum')(X)
        C3 = ContrastLoss(mode='way1', cdim=1, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        C1 = ContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction=None)(X)
        C2 = ContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='sum')(X)
        C3 = ContrastLoss(mode='way1', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(C1, C2, C3)


        # output
        tensor([[1.2612, 1.1085],
                [1.5992, 1.2124],
                [0.8201, 0.9887],
                [1.4376, 1.0091],
                [1.1397, 1.1860]]) tensor(11.7626) tensor(1.1763)
        tensor([0.6321, 1.1808, 0.5884, 1.1346, 0.6038]) tensor(4.1396) tensor(0.8279)
        tensor([0.6321, 1.1808, 0.5884, 1.1346, 0.6038]) tensor(4.1396) tensor(0.8279)

    """

    def __init__(self, mode='way1', cdim=None, dim=None, keepcdim=False, reduction='mean'):
        ...

    def forward(self, X):

