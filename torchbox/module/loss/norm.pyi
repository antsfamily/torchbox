class FnormLoss(th.nn.Module):
    r"""F-norm Loss

    Both complex and real representation are supported.

    .. math::
       {\rm norm}({\bf P}) = \|{\bf P}\|_2 = \left(\sum_{x_i\in {\bf P}}|x_i|^2\right)^{\frac{1}{2}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, None or optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    tensor
         the inputs's f-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        print('---norm')

        # real
        F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        # complex in real format
        F1 = FnormLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        F2 = FnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        F3 = FnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        F1 = FnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        F2 = FnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        F3 = FnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

class PnormLoss(th.nn.Module):
    r"""obtain the p-norm of a tensor

    Both complex and real representation are supported.

    .. math::
       {\rm pnorm}({\bf P}) = \|{\bf P}\|_p = \left(\sum_{x_i\in {\bf P}}|x_i|^p\right)^{\frac{1}{p}}

    where, :math:`u, v` are the real and imaginary part of x, respectively.

    Parameters
    ----------
    p : int
        Specifies the power. The default is 2.
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, None or optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')
    
    Returns
    -------
    tensor
         the inputs's p-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        print('---norm')

        # real
        F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        # complex in real format
        F1 = PnormLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        F2 = PnormLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        F3 = PnormLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(F1, F2, F3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        F1 = PnormLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        F2 = PnormLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        F3 = PnormLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
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

    def __init__(self, p=2, cdim=None, dim=None, keepdim=False, reduction='mean'):
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


