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
        The dimension axis for computing norm. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        Keep dimension?
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
        tensor([[3.0401, 4.9766],
                [4.8830, 3.1261],
                [6.3124, 4.1407],
                [5.9283, 4.5896],
                [3.4909, 6.7252]]) tensor(47.2130) tensor(4.7213)
        tensor([5.8317, 5.7980, 7.5493, 7.4973, 7.5772]) tensor(34.2535) tensor(6.8507)
        tensor([5.8317, 5.7980, 7.5493, 7.4973, 7.5772]) tensor(34.2535) tensor(6.8507)
    """

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
        The dimension axis for computing norm. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        Keep dimension?
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
        tensor([[3.0401, 4.9766],
                [4.8830, 3.1261],
                [6.3124, 4.1407],
                [5.9283, 4.5896],
                [3.4909, 6.7252]]) tensor(47.2130) tensor(4.7213)
        tensor([5.8317, 5.7980, 7.5493, 7.4973, 7.5772]) tensor(34.2535) tensor(6.8507)
        tensor([5.8317, 5.7980, 7.5493, 7.4973, 7.5772]) tensor(34.2535) tensor(6.8507)
    """


