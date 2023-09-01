def sl(dims, axis, idx=None, **kwargs):
    r"""Slice any axis

    generates slice in specified axis.

    Parameters
    ----------
    dims : int
        total dimensions
    axis : int or list
        select axis list.
    idx : list or None, optional
        slice lists of the specified :attr:`axis`, if None, does nothing (the default)
    dim : int or list
        (kwargs) if specified, will overwrite :attr:`axis`

    Returns
    -------
    tuple of slice
        slice for specified axis elements.

    Examples
    --------

    ::

        import numpy as np

        np.random.seed(2020)
        X = np.random.randint(0, 100, (9, 10))
        print(X, 'X)
        print(X[sl(2, -1, [0, 1])], 'Xsl')

        # output:

        [[96  8 67 67 91  3 71 56 29 48]
        [32 24 74  9 51 11 55 62 67 69]
        [48 28 20  8 38 84 65  1 79 69]
        [74 73 62 21 29 90  6 38 22 63]
        [21 68  6 98  3 20 55  1 52  9]
        [83 82 65 42 66 55 33 80 82 72]
        [94 91 14 14 75  5 38 83 99 10]
        [80 64 79 30 84 22 46 26 60 13]
        [24 63 25 89  9 69 47 89 55 75]] X
        [[96  8]
        [32 24]
        [48 28]
        [74 73]
        [21 68]
        [83 82]
        [94 91]
        [80 64]
        [24 63]] Xsl
    """

def cut(x, pos, axis=None, **kwargs):
    r"""Cut array at given position.

    Cut array at given position.

    Parameters
    ----------
    x : array or tensor
        a tensor to be cut
    pos : tuple or list
        cut positions: ((cpstart, cpend), (cpstart, cpend), ...)
    axis : int, tuple or list, optional
        cut axis (the default is None, which means nothing)
    """

def arraycomb(arrays, out=None):
    r"""compute the elemnts combination of several lists.

    Args:
        arrays (list or tensor): The lists or tensors.
        out (tensor, optional): The combination results (defaults is :obj:`None`).

    Returns:
        tensor: The combination results.

    Examples:

    Compute the combination of three lists: :math:`[1,2,3]`, :math:`[4, 5]`, :math:`[6,7]`,
    this will produce a :math:`12\times 3` array.

    ::

        x = arraycomb(([1, 2, 3], [4, 5], [6, 7]))
        print(x, x.shape)

        # output:
        [[1 4 6]
        [1 4 7]
        [1 5 6]
        [1 5 7]
        [2 4 6]
        [2 4 7]
        [2 5 6]
        [2 5 7]
        [3 4 6]
        [3 4 7]
        [3 5 6]
        [3 5 7]] (12, 3)

    """

def permute(X, dim, mode=None, dir='f'):
    """permutes axes of tensor

    Parameters
    ----------
    X : Tensor
        the input tensor
    dim : list or tuple
        the order of new dimensions (:attr:`mode` is :obj:`None`) or multiplication dimensions (``'matmul'``)
    mode : str or None, optional
        permution mode, ``'matmul'`` for matrix multiplication, 
        ``'merge'`` for dimension merging (putting the dimensions specified by second and subsequent elements of :attr:`dim`
        after the dimension specified by the specified by the first element of :attr:`dim`),
        :obj:`None` for regular permute, such as torch.permute, by default :obj:`None`.
    dir : str, optional
        the direction, ``'f'`` or ``'b'`` (reverse process of ``'f'``), default is ``'f'``.
    """    

def reduce(X, dim, keepdim, reduction):
    """reduce tensor in speciffied dimensions

    Parameters
    ----------
    X : Tensor
        the input tensor
    dim : int, list or tuple
        the dimensions for reduction
    keepdim : bool
        whether keep dimensions
    reduction : str or None
        The mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'``

    Returns
    -------
    tensor
        the reduced tensor

    Raises
    ------
    ValueError
        reduction mode
    """

def swap(x, dim1, dim2):
    """swap dimensions of input

    Parameters
    ----------
    x : Tensor
        the input
    dim1 : int, list or tuple
        the first dimension
    dim2 : int, list or tuple
        the first dimension

    Returns
    -------
    tensor
        the result

    Raises
    ------
    TypeError
        :attr:`dim1` and :attr:`dim2` must be integer, list or tuple.
    """

def merge(x, dim, keepdim=False):
    """merge tensor's dimensions

    Parameters
    ----------
    x : Tensor
        the input
    dim : int, list or tuple
        dimensions indexes for merging, putting the dimensions specified by second and subsequent elements of :attr:`dim`
        after the dimension specified by the specified by the first element of :attr:`dim`)
    keepdim : bool, optional
        keep the dimensions?, by default False

    Returns
    -------
    tensor
        _description_
    """


