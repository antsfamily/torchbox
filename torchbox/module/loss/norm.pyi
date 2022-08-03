class FnormLoss(th.nn.Module):
    r"""F-norm Loss

    Both complex and real representation are supported.

    .. math::
       {\rm norm}({\bf X}) = \|{\bf X}\|_2 = \left(\sum_{x_i\in {\bf X}}|x_i|^2\right)^{\frac{1}{2}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : tensor
        input
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, None or optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    tensor
         the inputs's f-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        print('---norm')

        # real
        F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
        F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        # complex in real format
        F1 = FnormLoss(cdim=1, dim=(-2, -1), reduction=None)(X, Y)
        F2 = FnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = FnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
        F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        ---norm
        tensor([[2.8719, 2.8263],
                [3.1785, 3.4701],
                [4.6697, 3.2955],
                [3.0992, 2.6447],
                [3.5341, 3.5779]]) tensor(33.1679) tensor(3.3168)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
    """

    def __init__(self, cdim=None, dim=None, keepcdim=False, reduction='mean'):
        ...

    def forward(self, X, Y):
        ...

class PnormLoss(th.nn.Module):
    r"""obtain the p-norm of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm pnorm}({\bf X}) = \|{\bf X}\|_p = \left(\sum_{x_i\in {\bf X}}|x_i|^p\right)^{\frac{1}{p}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    X : tensor
        input
    p : int
        Specifies the power. The default is 2.
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing norm. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, None or optional
        The operation in batch dim, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')
    
    Returns
    -------
    tensor
         the inputs's p-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        print('---norm')

        # real
        F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
        F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        # complex in real format
        F1 = PnormLoss(cdim=1, dim=(-2, -1), reduction=None)(X, Y)
        F2 = PnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = PnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(X, Y)
        F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(X, Y)
        F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(X, Y)
        print(F1, F2, F3)

        ---norm
        tensor([[2.8719, 2.8263],
                [3.1785, 3.4701],
                [4.6697, 3.2955],
                [3.0992, 2.6447],
                [3.5341, 3.5779]]) tensor(33.1679) tensor(3.3168)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
        tensor([4.0294, 4.7058, 5.7154, 4.0743, 5.0290]) tensor(23.5539) tensor(4.7108)
    """

    def __init__(self, p=2, cdim=None, dim=None, keepcdim=False, reduction='mean'):
        ...

    def forward(self, X, Y):

