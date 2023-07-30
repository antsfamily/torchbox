def rdcdim(ndim, cdim, dim, keepcdim=False, reduction=None):
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

def rmcdim(ndim, dim, cdim, keepdim):
    r"""re-define dimensions

    Parameters
    ----------
    ndim : int
        the number of dimensions
    dim : int, None, tuple or list
        dimensions to be re-defined
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to :obj:`None`
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)

    Returns
    -------
    int, tuple or list
         re-defined dimensions
        
    """

def reduce(X, dim, keepdim, reduction):
    """reduce tensor in speciffied dimensions

    Parameters
    ----------
    X : tensor
        the input tensor
    dim : list or tuple
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


