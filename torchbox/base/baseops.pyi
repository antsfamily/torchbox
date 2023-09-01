def dimpos(ndim, dim):
    """make positive dimensions

    Parameters
    ----------
    ndim : int
        the number of dimensions
    dim : int, list or tuple
        the dimension index to be converted
    """

def rmcdim(ndim, cdim, dim, keepdim):
    r"""get dimension indexes after removing cdim

    Parameters
    ----------
    ndim : int
        the number of dimensions
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to :obj:`None`
    dim : int, None, tuple or list
        dimensions to be re-defined
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)

    Returns
    -------
    int, tuple or list
         re-defined dimensions
        
    """

def dimpermute(ndim, dim, mode=None, dir='f'):
    """permutes dimensions

    Parameters
    ----------
    ndim : int
        the number of dimensions
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

def dimreduce(ndim, cdim, dim, keepcdim=False, reduction=None):
    """get dimensions for reduction operation

    Parameters
    ----------
    ndim : int
        the number of dimensions
    cdim : int, optional
        if the data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to :obj:`None`
    dim : int, list, tuple or None
        dimensions for processing, :obj:`None` means all
    keepcdim : bool
        keep the complex dimension? The default is :obj:`False`
    reduction : str or None, optional
        The operation in other dimensions except the dimensions specified by :attr:`dim`,
        ``None``, ``'mean'`` or ``'sum'`` (the default is :obj:`None`)

    """    

def dimmerge(ndim, mdim, dim, keepdim=False):
    """obtain new dimension indexes after merging

    Parameters
    ----------
    ndim : int
        the number of dimensions
    mdim : int, list or tuple
        the dimension indexes for merging
    dim : int, list or tuple
        the dimension indexes that are not merged
    keepdim : bool
        keep the dimensions when merging?
    """

def upkeys(D, mode='-', k='module.'):
    r"""update keys of a dictionary

    Parameters
    ----------
    D : dict
        the input dictionary
    mode : str, optional
        ``'-'`` for remove key string which is specified by :attr:`k`, by default '-'
        ``'+'`` for add key string which is specified by :attr:`k`, by default '-'
    k : str, optional
        key string pattern, by default 'module.'

    Returns
    -------
    dict
        new dictionary with keys updated
    """

def dreplace(d, fv=None, rv='None', new=False):
    """replace dict value

    Parameters
    ----------
    d : dict
        the dict
    fv : any, optional
        to be replaced, by default None
    rv : any, optional
        replaced with, by default 'None'
    new : bool, optional
        if true, deep copy dict, will not change input, by default False

    Returns
    -------
    dict
        dict with replaced value
    """

def dmka(D, Ds):
    r"""Multiple key-value assign to a dict

    Parameters
    ----------
    D : dict
        main dict
    Ds : dict
        sub dict

    Returns
    -------
    dict
        after assign
    """

def cat(shapes, axis=0):
    r"""Concatenates

    Concatenates the given sequence of seq shapes in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    Parameters
    ----------
    shapes : tuples or lists
        (shape1, shape2, ...)
    axis : int, optional
        specify the concatenated axis (the default is 0)

    Returns
    -------
    tuple or list
        concatenated shape

    Raises
    ------
    ValueError
        Shapes are not consistent in axises except the specified one.
    """

def argsort(x, reverse=False):
    r"""returns index of sorted array

    Parameters
    ----------
    x : list, ndarray or tensor
        the input
    reverse : bool, optional
        sort in reversed order?, by default False

    Returns
    -------
    list, ndarray or tensor
        the index
    """


