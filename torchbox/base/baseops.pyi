def redim(ndim, dim, cdim, keepcdim):
    r"""re-define dimensions

    Parameters
    ----------
    ndim : int
        the number of dimensions
    dim : int, tuple or list
        dimensions to be re-defined
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to None, defaults is None.
        For example, :math:`{\bm X}_c\in {\mathbb C}^{N\times C\times H\times W}` is
        represented as a real-valued tensor :math:`{\bm X}_r\in {\mathbb R}^{N\times C\times H\times W\ times 2}`,
        then :attr:`cdim` equals to -1 or 4.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.

    Returns
    -------
    int, tuple or list
         re-defined dimensions
        
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


