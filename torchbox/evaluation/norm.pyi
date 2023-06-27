def norm(X, mode='2', cdim=None, dim=None, keepdim=False, reduction=None):
    r"""obtain the norm of a tensor

    Both complex and real representation are supported.

    F-norm (Frobenius):

    .. math::
       \|{\bf X}\|_F = \|{\bf X}\|_p = \left(\sum_{x_i\in {\bf X}}|x_i|^2\right)^{\frac{1}{2}}
    
    p-norm:

    .. math::
       \|{\bf X}\|_p = \|{\bf X}\|_p = \left(\sum_{x_i\in {\bf X}}|x_i|^p\right)^{\frac{1}{p}}

    2-norm or spectral norm:

    .. math::
       \|{\bf X}\|_2 = \sqrt{\lambda_1} = \sqrt{{\rm max} {\lambda({\bf X}^H{\bf X})}}

    1-norm:
    
    .. math::
       \|{\bf X}\|_1 = {\rm max}\sum_{i=1}^M|x_ij|

       
    Parameters
    ----------
    X : tensor
        input
    mode : str
        the mode of norm. ``'2'`` means 2-norm (default), ``'1'`` means 1-norm, ``'px'`` means p-norm (x is the power), 
        ``'fro'`` means Frobenius-norm  The default is ``'2'``.
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : int or None
        The dimension axis for computing norm. For 2-norm, :attr:`dim` must be specified. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str, optional
        The operation in batch dim, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    tensor
         the inputs's p-norm.

    Examples
    ---------

    ::

        th.manual_seed(2020)
        X, cdim = th.randn(5, 2, 3, 4), 1
        X, cdim = th.randn(2, 3, 4), 0

        # real
        C1 = norm(X, mode='fro', cdim=None, dim=(-2, -1), keepdim=False)
        C2 = norm(X, mode='2', cdim=None, dim=(-2, -1), keepdim=False)
        C3 = norm(X, mode='1', cdim=None, dim=-1, keepdim=False)
        C4 = norm(X, mode='p1', cdim=None, dim=(-2, -1), keepdim=False)
        C5 = norm(X, mode='p2', cdim=None, dim=(-2, -1), keepdim=False)
        print(C1, C2, C3, C4, C5)

        # complex in real format
        C1 = norm(X, mode='fro', cdim=cdim, dim=(-2, -1), keepdim=False)
        C2 = norm(X, mode='2', cdim=cdim, dim=(-2, -1), keepdim=False)
        C3 = norm(X, mode='1', cdim=cdim, dim=-1, keepdim=False)
        C4 = norm(X, mode='p1', cdim=cdim, dim=(-2, -1), keepdim=False)
        C5 = norm(X, mode='p2', cdim=cdim, dim=(-2, -1), keepdim=False)
        print(C1, C2, C3, C4, C5)

        # complex in complex format
        X = tb.r2c(X, cdim=cdim, keepdim=False)
        C1 = norm(X, mode='fro', cdim=None, dim=(-2, -1), keepdim=False)
        C2 = norm(X, mode='2', cdim=None, dim=(-2, -1), keepdim=False)
        C3 = norm(X, mode='1', cdim=None, dim=-1, keepdim=False)
        C4 = norm(X, mode='p1', cdim=None, dim=(-2, -1), keepdim=False)
        C5 = norm(X, mode='p2', cdim=None, dim=(-2, -1), keepdim=False)
        print(C1, C2, C3, C4, C5)

        # ---output
        tensor([2.0562, 3.8482]) tensor([ 2.5458, 10.1084]) tensor([[0.5087, 1.1792, 0.9083],
                [2.2781, 1.3459, 1.0774]]) tensor([ 5.6931, 10.5262]) tensor([2.0562, 3.8482])
        tensor(4.3631) tensor(11.2836) tensor([2.2842, 1.4056, 1.0787]) tensor(13.4182) tensor(4.3631)
        tensor(4.3631) tensor(11.2836) tensor([2.2842, 1.4056, 1.0787]) tensor(13.4182) tensor(4.3631)
    """


