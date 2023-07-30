def cossim(P, G, mode=None, cdim=None, dim=None, keepdim=False, reduction=None):
    r"""compute cosine similarity

    .. math::
       s = \frac{<{\bf p}, {\bf g}>}{\|p\|_2\|g\|_2}

    .. note:: 
       For complex, the magnitude still gives the "similarity" between them, 
       where the complex angle gives the complex phase factor required to fully reach that similarity. 
       refers `Cosine similarity between complex vectors <https://math.stackexchange.com/questions/273527/cosine-similarity-between-complex-vectors#:~:text=For%20complex%2C%20the%20magnitude%20still%20gives%20the%20%22similarity%22,is%20a%20real%20scalar%20multiple%20of%20y%20y.>`_

    Parameters
    ----------
    P : tensor
        the first/left input, such as the predicted
    G : tensor
        the second/right input, such as the ground-truth
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : tuple, None, optional
        The dimension axis for computing cosine similarity. The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    S : scalar or tensor
        The cosine similarity value of inputs.
    
    Examples
    --------

    ::

        import torch as th
        from torchbox import cossim

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)
        dim = (-2, -1)

        # real
        C1 = cossim(X, Y, cdim=None, dim=dim, reduction=None)
        C2 = cossim(X, Y, cdim=None, dim=dim, reduction='sum')
        C3 = cossim(X, Y, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = cossim(X, Y, cdim=1, dim=dim, reduction=None)
        C2 = cossim(X, Y, cdim=1, dim=dim, reduction='sum')
        C3 = cossim(X, Y, cdim=1, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = cossim(X, Y, cdim=None, dim=dim, reduction=None)
        C2 = cossim(X, Y, cdim=None, dim=dim, reduction='sum')
        C3 = cossim(X, Y, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

    """

def peacor(P, G, mode=None, cdim=None, dim=None, keepdim=False, reduction=None):
    r"""compute the Pearson Correlation Coefficient

    The Pearson correlation coefficient can be viewed as the cosine similarity of centered (remove mean) input.

    Parameters
    ----------
    P : tensor
        the first/left input, such as the predicted
    G : tensor
        the second/right input, such as the ground-truth
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`X` is complex-valued, :attr:`cdim` is ignored. If :attr:`X` is real-valued and :attr:`cdim` is integer
        then :attr:`X` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`X` will be treated as real-valued
    dim : tuple, None, optional
        The dimension axis for computing the Pearson correlation coefficient. 
        The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    S : scalar or tensor
        The Pearson correlation coefficient value of inputs.
    
    Examples
    --------

    ::

        import torch as th
        from torchbox import peacor

        th.manual_seed(2020)
        X = th.randn(5, 2, 3, 4)
        Y = th.randn(5, 2, 3, 4)
        dim = (-2, -1)

        # real
        C1 = peacor(X, Y, cdim=None, dim=dim, reduction=None)
        C2 = peacor(X, Y, cdim=None, dim=dim, reduction='sum')
        C3 = peacor(X, Y, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = peacor(X, Y, cdim=1, dim=dim, reduction=None)
        C2 = peacor(X, Y, cdim=1, dim=dim, reduction='sum')
        C3 = peacor(X, Y, cdim=1, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        X = X[:, 0, ...] + 1j * X[:, 1, ...]
        Y = Y[:, 0, ...] + 1j * Y[:, 1, ...]
        C1 = peacor(X, Y, cdim=None, dim=dim, reduction=None)
        C2 = peacor(X, Y, cdim=None, dim=dim, reduction='sum')
        C3 = peacor(X, Y, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        x = th.randn(2, 3) + 1j*th.randn(2, 3)
        print(th.corrcoef(x))
        print(peacor(x[0], x[1]))
        print(peacor(x[1], x[0]))

    """


