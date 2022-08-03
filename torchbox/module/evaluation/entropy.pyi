class Entropy(th.nn.Module):
    r"""compute the entropy of the inputs

    .. math::
        {\rm S} = -\sum_{n=0}^N p_i{\rm log}_2 p_n

    where :math:`N` is the number of pixels, :math:`p_n=\frac{|X_n|^2}{\sum_{n=0}^N|X_n|^2}`.

    Parameters
    ----------
    X : tensor
        The complex or real inputs, for complex inputs, both complex and real representations are surpported.
    mode : str, optional
        The entropy mode: ``'shannon'`` or ``'natural'`` (the default is 'shannon')
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis (if :attr:`keepcdim` is :obj:`False` then :attr:`cdim` is not included) for computing entropy. 
        The default is :obj:`None`, which means all. 
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    reduction : str, optional
        The operation in batch dim, ``'None'``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : tensor
        The entropy of the inputs.
    
    Examples
    --------

    ::

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)

        # real
        S1 = Entropy(mode='shannon', cdim=None, dim=(-2, -1), reduction=None)(X)
        S2 = Entropy(mode='shannon', cdim=None, dim=(-2, -1), reduction='sum')(X)
        S3 = Entropy(mode='shannon', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(S1, S2, S3)

        # complex in real format
        S1 = Entropy(mode='shannon', cdim=1, dim=(-2, -1), reduction=None)(X)
        S2 = Entropy(mode='shannon', cdim=1, dim=(-2, -1), reduction='sum')(X)
        S3 = Entropy(mode='shannon', cdim=1, dim=(-2, -1), reduction='mean')(X)
        print(S1, S2, S3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        S1 = Entropy(mode='shannon', cdim=None, dim=(-2, -1), reduction=None)(X)
        S2 = Entropy(mode='shannon', cdim=None, dim=(-2, -1), reduction='sum')(X)
        S3 = Entropy(mode='shannon', cdim=None, dim=(-2, -1), reduction='mean')(X)
        print(S1, S2, S3)

        # output
        tensor([[2.5482, 2.7150],
                [2.0556, 2.6142],
                [2.9837, 2.9511],
                [2.4296, 2.7979],
                [2.7287, 2.5560]]) tensor(26.3800) tensor(2.6380)
        tensor([3.2738, 2.5613, 3.2911, 2.7989, 3.2789]) tensor(15.2040) tensor(3.0408)
        tensor([3.2738, 2.5613, 3.2911, 2.7989, 3.2789]) tensor(15.2040) tensor(3.0408)
    """

    def __init__(self, mode='shannon', cdim=None, dim=None, keepcdim=False, reduction='mean'):
        ...

    def forward(self, X):

