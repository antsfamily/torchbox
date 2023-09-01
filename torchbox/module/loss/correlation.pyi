class CosSimLoss(th.nn.Module):
    r"""compute the cosine similarity loss of the inputs

    If utilize the amplitude of correlation as loss

    .. math::
       {\mathcal L} = 1 - |\frac{<{\bf p}, {\bf g}>}{\|{\bf p}\|_2\|{\bf g}\|_2}|

    If utilize the angle of correlation as loss
    
    .. math::
       {\mathcal L} = |\angle \frac{<{\bf p}, {\bf g}>}{\|{\bf p}\|_2\|{\bf g}\|_2}|

    Parameters
    ----------
    mode : str
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity.
    cdim : int or None
        If :attr:`P` and :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` and :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`P` and :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` and :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing correlation. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : Tensor
        The correlation of the inputs.

    see also :func:`~torchbox.evaluation.correlation.cossim`, :func:`~torchbox.evaluation.correlation.peacor`, :func:`~torchbox.evaluation.correlation.eigveccor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.PeaCorLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.
    
    Examples
    --------

    ::

        import torch as th
        from torchbox import CosSimLoss

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        S1 = CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        S2 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        S3 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in real format
        S1 = CosSimLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        S2 = CosSimLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        S3 = CosSimLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        S1 = CosSimLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        S2 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        S3 = CosSimLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # output
        tensor([[0.4791, 0.0849],
                [0.0334, 0.4855],
                [0.0136, 0.2280],
                [0.4951, 0.2166],
                [0.4484, 0.4221]]) tensor(2.9068) tensor(0.2907)
        tensor([[0.2926],
                [0.2912],
                [0.1505],
                [0.3993],
                [0.3350]]) tensor([1.4685]) tensor([0.2937])
        tensor([0.2926, 0.2912, 0.1505, 0.3993, 0.3350]) tensor(1.4685) tensor(0.2937)
    
    """

    def __init__(self, mode='abs', cdim=None, dim=None, keepdim=False, reduction='mean'):
        ...

    def forward(self, P, G):
        """forward process

        Parameters
        ----------
        P : Tensor
            predicted/estimated/reconstructed
        G : Tensor
            ground-truth/target

        Returns
        -------
        S : Tensor
            The correlation of the inputs.
        """

class PeaCorLoss(th.nn.Module):
    r"""compute the pearson correlation loss of the inputs

    If utilize the amplitude of pearson correlation as loss

    .. math::
       {\mathcal L} = 1 - |\frac{<{\bf p}, {\bf g}>}{\|{\bf p}\|_2\|{\bf g}\|_2}|

    If utilize the angle of pearson correlation as loss

    .. math::
       {\mathcal L} = |\angle \frac{<{\bf p}, {\bf g}>}{\|{\bf p}\|_2\|{\bf g}\|_2}|
       
    where :math:`\bf p` and :math:`\bf g` is the centered version (removed mean) of inputs

    Parameters
    ----------
    mode : str
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity.
    cdim : int or None
        If :attr:`P` and :attr:`G` is complex-valued, :attr:`cdim` is ignored. If :attr:`P` and :attr:`G` is real-valued and :attr:`cdim` is integer
        then :attr:`P` and :attr:`G` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`P` and :attr:`G` will be treated as real-valued
    dim : int or None
        The dimension axis for computing correlation. 
        The default is :obj:`None`, which means all. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : Tensor
        The correlation of the inputs.
    
    see also :func:`~torchbox.evaluation.correlation.cossim`, :func:`~torchbox.evaluation.correlation.peacor`, :func:`~torchbox.evaluation.correlation.eigveccor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.CosSimLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.
    
    Examples
    --------

    ::

        import torch as th
        from torchbox import PeaCorLoss

        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        S1 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        S2 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        S3 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in real format
        S1 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction=None)(P, G)
        S2 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction='sum')(P, G)
        S3 = PeaCorLoss(cdim=1, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        S1 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction=None)(P, G)
        S2 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='sum')(P, G)
        S3 = PeaCorLoss(cdim=None, dim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)


        # output
        tensor([[0.6010, 0.0260],
                [0.0293, 0.4981],
                [0.0063, 0.2284],
                [0.3203, 0.2851],
                [0.3757, 0.3936]]) tensor(2.7639) tensor(0.2764)
        tensor([[0.3723],
                [0.2992],
                [0.1267],
                [0.3020],
                [0.2910]]) tensor([1.3911]) tensor([0.2782])
        tensor([0.3723, 0.2992, 0.1267, 0.3020, 0.2910]) tensor(1.3911) tensor(0.2782)
    
    """

    def __init__(self, mode='abs', cdim=None, dim=None, keepdim=False, reduction='mean'):
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

class EigVecCorLoss(th.nn.Module):
    r"""compute the eigenvector correlation of the inputs

    Parameters
    ----------
    mode : str
        only work when :attr:`P` and :attr:`G` are complex-valued in real format or complex format.
        ``'abs'`` or ``'amplitude'`` returns the amplitude of similarity, ``'angle'`` or ``'phase'`` returns the phase of similarity.
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
        The operation mode of reduction, ``None``, ``'mean'`` or ``'sum'`` (the default is 'mean')

    Returns
    -------
    S : Tensor
        The eigenvector correlation of the inputs.
    
    see also :func:`~torchbox.evaluation.correlation.cossim`, :func:`~torchbox.evaluation.correlation.peacor`, :func:`~torchbox.evaluation.correlation.eigveccor`, :obj:`~torchbox.module.evaluation.correlation.CosSim`, :obj:`~torchbox.module.evaluation.correlation.PeaCor`, :obj:`~torchbox.module.evaluation.correlation.EigVecCor`, :obj:`~torchbox.module.loss.correlation.CosSimLoss`, :obj:`~torchbox.module.loss.correlation.EigVecCorLoss`.

    Examples
    --------

    ::

        import torch as th
        from torchbox import EigVecCorLoss

        mode = 'abs'
        th.manual_seed(2020)
        P = th.randn(5, 2, 3, 4)
        G = th.randn(5, 2, 3, 4)

        # real
        S1 = EigVecCorLoss(npcs=4, mode=mode, cdim=None, sdim=0, fdim=(-2, -1), reduction=None)(P, G)
        S2 = EigVecCorLoss(npcs=4, mode=mode, cdim=None, sdim=0, fdim=(-2, -1), reduction='sum')(P, G)
        S3 = EigVecCorLoss(npcs=4, mode=mode, cdim=None, sdim=0, fdim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in real format
        S1 = EigVecCorLoss(npcs=4, mode=mode, cdim=1, sdim=0, fdim=(-2, -1), reduction=None)(P, G)
        S2 = EigVecCorLoss(npcs=4, mode=mode, cdim=1, sdim=0, fdim=(-2, -1), reduction='sum')(P, G)
        S3 = EigVecCorLoss(npcs=4, mode=mode, cdim=1, sdim=0, fdim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)

        # complex in complex format
        P = P[:, 0, ...] + 1j * P[:, 1, ...]
        G = G[:, 0, ...] + 1j * G[:, 1, ...]
        S1 = EigVecCorLoss(npcs=4, mode=mode, cdim=None, sdim=0, fdim=(-2, -1), reduction=None)(P, G)
        S2 = EigVecCorLoss(npcs=4, mode=mode, cdim=None, sdim=0, fdim=(-2, -1), reduction='sum')(P, G)
        S3 = EigVecCorLoss(npcs=4, mode=mode, cdim=None, sdim=0, fdim=(-2, -1), reduction='mean')(P, G)
        print(S1, S2, S3)
    
    """

    def __init__(self, npcs=4, mode=None, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction='mean'):
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


