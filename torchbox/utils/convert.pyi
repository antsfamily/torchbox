def str2hash(s, hmode='sha256', enc='utf-8', tohex=True):
    r"""convert a string to hash code

    Args:
        s (str): the input
        hmode (str or hash function, optional): must either be a hash algorithm name as a str, a hash constructor, or a callable that returns a hash object. ``'sha1'``, ``'sha224'``, ``'sha256'``, ``'sha384'``, ``'sha512'``, ``'md5'``, ..., see `hashlib <https://docs.python.org/3/library/hashlib.html?highlight=new#module-hashlib>`_ Defaults to 'sha256'.
        tohex (bool, optional): return hex code? Defaults to :obj:`True`.
    """

def file2hash(file, hmode='sha256', tohex=True):
    r"""convert a string to hash code

    Args:
        s (str): the input
        hmode (str or hash function, optional): must either be a hash algorithm name as a str, a hash constructor, or a callable that returns a hash object. ``'sha1'``, ``'sha224'``, ``'sha256'``, ``'sha384'``, ``'sha512'``, ``'md5'``, ..., see `hashlib <https://docs.python.org/3/library/hashlib.html?highlight=new#module-hashlib>`_ Defaults to 'sha256'.
        tohex (bool, optional): return hex code? Defaults to :obj:`True`.
    """

def dict2str(ddict, attrtag=': ', indent='  ', linebreak='\n', nindent=0):
    r"""dump dict object to str

    Parameters
    ----------
    ddict : dict
        The dict object to be converted
    attrtag : str, optional
        The tag of attribution, by default ``': '``
    indent : str, optional
        The dict identifier, by default ``'  '``
    linebreak : str, optional
        The line break character, by default '\n'
    nindent : int, optional
        the number of initial indent characters, by default 0

    Returns
    -------
    str
        The converted string.
    """

def str2bool(s, truelist=['true', '1', 'y', 'yes'], falselist=['false', '0', 'n', 'no']):
    r"""Converts string to bool

    Parameters
    ----------
    s : str
        The input string
    truelist : list
        The true flag list.
    falselist : list
        The false flag list.
    """

def str2list(s, sep=' '):
    r"""Converts string to list

    Parameters
    ----------
    s : str
        The input string
    sep : str
        The separator, only work when :func:`literal_eval` fails.

    Examples
    --------

    ::

        s = '[0, [[[[1], 2.], 33], 4], [5, [6, 2.E-3]], 7, [8]], 1e-3'
        print(str2list(s))

        # ---output
        ([0, [[[[1], 2.0], 33], 4], [5, [6, 0.002]], 7, [8]], 0.001)


    """

def str2tuple(s, sep=' '):
    r"""Converts string to tuple

    Parameters
    ----------
    s : str
        The input string
    sep : str
        The separator, only work when :func:`literal_eval` fails.

    Examples
    --------

    ::

        s = '[0, [[[[1], 2.], 33], 4], [5, [6, 2.E-3]], 7, [8]], 1e-3'
        print(str2list(s))

        # ---output
        ([0, [[[[1], 2.0], 33], 4], [5, [6, 0.002]], 7, [8]], 0.001)


    """

def str2num(s, vfn=None):
    r"""Extracts numbers in a string.

    Parameters
    ----------
    s : str
        The string.
    vfn : None, optional
        formating function, such as ``int``, ``float`` or ``'auto'``.

    Returns
    -------
    list
        The number list.

    Examples
    --------

    ::

        print(str2num(s, int))
        print(str2num(s, float))
        print(str2num(s, 'auto'))

        print(2**(str2num('int8', int)[0]))
        print(str2num('int', int) == [])
        
        # ---output
        [0, 1, 2, 33, 4, 5, 6, 0, 7, 8, 0]
        [0.0, 1.0, 2.0, 33.0, 4.0, 5.0, 6.0, 0.002, 7.0, 8.0, 0.001]
        [0, 1, 2.0, 33, 4, 5, 6, 0.002, 7, 8, 0.001]
        256
        True
    """

def str2sec(x, sep=':'):
    r"""Extracts second in a time string.
        
        ``hh:mm:ss``  -->  ``hh*3600 + mm*60 + ss``

    Parameters
    ----------
    s : str
        The string or string list/tuple.
    sep : str
        The separator between hour, minute and seconds, default is ``':'``.

    Returns
    -------
    y : int
        The seconds.

    Examples
    --------

    ::

        print(str2sec('1:00:0'))
        print(str2sec('1:10:0'))
        print(str2sec('1:10:6'))
        print(str2sec('1:10:30'))
        
        # ---output
        3600
        4200
        4206
        4230
    """

def int2bstr(n, nbytes, endian='<', signed=True):
    r"""converts integer to bytes string

    Parameters
    ----------
    n : int
        the input integer
    nbytes : int
        the number of bytes
    endian : str, optional
        byte order, supported are little endian: ``'<'`` (the default), big endian: ``'>'``.
    signed : bool, optional
        signed or unsigned, by default True

    Returns
    -------
    bstr
        The integer in binary string format.

    Examples
    --------

    ::

        n = -123

        bs = int2bstr(n, 4, '<', signed=True)
        print(bs)
        print(hex(n))
        print(bstr2int(bs, '<'))

        bs = int2bstr(n, 4, '>', signed=True)
        print(bs)
        print(hex(n))
        print(bstr2int(bs, '>'))

        # ---output
        b'\x85\xff\xff\xff'
        -0x7b
        -123
        b'\xff\xff\xff\x85'
        -0x7b
        -123

    """    

def bstr2int(b, endian='<', signed=True):
    r"""convert binary string data to integer

    Parameters
    ----------
    b : bstr
        an integer in binary format
    endian : str, optional
        The order of the bytes, supported are little endian: ``'<'`` (the default), big endian: ``'>'``.
    signed : bool, optional
        signed or unsigned, by default True
    
    Returns
    -------
    int
        The integer in decimal.

    Examples
    --------

    ::

        n = -123

        bs = int2bstr(n, 4, '<', signed=True)
        print(bs)
        print(hex(n))
        print(bstr2int(bs, '<'))

        bs = int2bstr(n, 4, '>', signed=True)
        print(bs)
        print(hex(n))
        print(bstr2int(bs, '>'))

        # ---output
        b'\x85\xff\xff\xff'
        -0x7b
        -123
        b'\xff\xff\xff\x85'
        -0x7b
        -123

    """


