def cossim(P, G, mode=None, cdim=None, dim=None, keepdim=False, reduction=None):
    r"""compute cosine similarity

    .. math::
       s = \frac{<{\bf p}, {\bf g}>}{\|{\bf p}\|_2\|{\bf g}\|_2}

    .. note:: 
       For complex, the magnitude still gives the "similarity" between them, 
       where the complex angle gives the complex phase factor required to fully reach that similarity. 
       refers `Cosine similarity between complex vectors <https://math.stackexchange.com/questions/273527/cosine-similarity-between-complex-vectors#:~:text=For%20complex%2C%20the%20magnitude%20still%20gives%20the%20%22similarity%22,is%20a%20real%20scalar%20multiple%20of%20y%20y.>`_

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
    dim : tuple, None, optional
        The dimension indexes for computing cosine similarity. The default is :obj:`None`, which means all.
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    S : scalar or tensor
        The cosine similarity value of inputs.
    
    see also :func:`~torchbox.evaluation.correlation.peacor`, :func:`~torchbox.evaluation.correlation.eigveccor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.CosSimLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.

    Examples
    --------

    ::

        import torch as th
        from torchbox import cossim

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)
        dim = (-2, -1)

        # real
        C1 = cossim(P, G, cdim=None, dim=dim, reduction=None)
        C2 = cossim(P, G, cdim=None, dim=dim, reduction='sum')
        C3 = cossim(P, G, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = cossim(P, G, cdim=1, dim=dim, reduction=None)
        C2 = cossim(P, G, cdim=1, dim=dim, reduction='sum')
        C3 = cossim(P, G, cdim=1, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = cossim(P, G, cdim=None, dim=dim, reduction=None)
        C2 = cossim(P, G, cdim=None, dim=dim, reduction='sum')
        C3 = cossim(P, G, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

    """

def peacor(P, G, mode=None, cdim=None, dim=None, keepdim=False, reduction=None):
    r"""compute the Pearson Correlation Coefficient

    The Pearson correlation coefficient can be viewed as the cosine similarity of centered (remove mean) input.

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`P` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` is real-valued and :attr:`cdim` is integer
        then :attr:`P` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` will be treated as real-valued
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

    see also :func:`~torchbox.evaluation.correlation.cossim`, :func:`~torchbox.evaluation.correlation.eigveccor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.CosSimLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.

    Examples
    --------

    ::

        import torch as th
        from torchbox import peacor

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)
        dim = (-2, -1)

        # real
        C1 = peacor(P, G, cdim=None, dim=dim, reduction=None)
        C2 = peacor(P, G, cdim=None, dim=dim, reduction='sum')
        C3 = peacor(P, G, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in real format
        C1 = peacor(P, G, cdim=1, dim=dim, reduction=None)
        C2 = peacor(P, G, cdim=1, dim=dim, reduction='sum')
        C3 = peacor(P, G, cdim=1, dim=dim, reduction='mean')
        print(C1, C2, C3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        C1 = peacor(P, G, cdim=None, dim=dim, reduction=None)
        C2 = peacor(P, G, cdim=None, dim=dim, reduction='sum')
        C3 = peacor(P, G, cdim=None, dim=dim, reduction='mean')
        print(C1, C2, C3)

        x = th.randn(2, 3) + 1j*th.randn(2, 3)
        print(th.corrcoef(x))
        print(peacor(x[0], x[1]))
        print(peacor(x[1], x[0]))

    """

def eigveccor(P, G, npcs=4, mode=None, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction=None):
    r"""computes cosine similarity of eigenvectors

    Parameters
    ----------
    P : Tensor
        predicted/estimated/reconstructed
    G : Tensor
        ground-truth/target
    npcs : int, optional
        the number principal components for comparing, by default 4
    mode : str or None
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity
        :obj:`None` returns the complex-valued similarity (default).
    cdim : int or None
        If :attr:`P` and :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` and :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`P` and :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` and :attr:`G` will be treated as real-valued
    sdim : int, optional
        the dimension index of sample, by default -1
    fdim : int, optional
        the dimension index of feature, by default -2
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    Returns
    -------
    S : scalar or tensor
        The eigenvector correlation value of inputs.
    
    see also :func:`~torchbox.evaluation.correlation.cossim`, :func:`~torchbox.evaluation.correlation.peacor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.CosSimLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.

    Examples
    --------

    ::

        import torch as th
        from torchbox import eigveccor
    
        print('---compare eigen vector correlation (complex in real)')
        G = th.randn(2, 3, 2, 64, 4)
        P = th.randn(2, 3, 2, 64, 4)
        print(eigveccor(G, G, npcs=4, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))
        print(eigveccor(G, G, npcs=4, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction=None).shape)
        
        print('---compare eigen vector correlation (complex in complex)')
        G = th.randn(2, 3, 64, 4) + 1j*th.randn(2, 3, 64, 4)
        P = th.randn(2, 3, 64, 4) + 1j*th.randn(2, 3, 64, 4)
        print(eigveccor(G, G, npcs=4, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))
        print(eigveccor(G, G, npcs=4, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction=None).shape)
        
    """


