def data_path(dsname=None):
    r"""obtain data source path

    Parameters
    ----------
    dsname : str, optional
        dataset name (``'character'``, ``'shape'``, ``'optical'``, ``'remote'``, ``'mstar'``), by default None

    Returns
    -------
    str
        the path string.
    """    

def pkg_path():
    """obtain this package's path

    Returns
    -------
    str
        package path string.
    """    

def copyfile(srcfile, dstfile):
    r"""copy file

    copy file from srcfile to dstfile.

    Parameters
    ----------
    srcfile : str
        the source file path string
    dstfile : str
        the destnation file path string
    """

def copyfiles(srcdir, dstdir, filenames):
    r"""copy files

    copy files from source directory to destnation directory

    Parameters
    ----------
    srcdir : str
        the source directory path string
    dstdir : str
        the destnation directory path string
    filenames : list or tuple
        filenames to be copied
    """    

def __listxfile__(listdir=None, exts=None, recursive=False, filelist=[]):
    ...

def listxfile(listdir=None, exts=None, recursive=False):
    r"""List the files in a directory.


    Parameters
    ----------
    listdir : None, optional
        The directory for listing. The default is None, which means return :attr:`filelist` directly.
    exts : str, list or None, optional
        File extension string or extension string list, such as ``'.png'`` or ``['.png', 'jpg']``.
        The default is None, which means any extension.
    recursive : bool, optional
        Recursive search? The default is False, which means only list the root directory (:attr:`listdir`).

    Returns
    -------
    list
        The list of file path with extension :attr:`exts`. Sometimes you need to sort the list using ``sorted``. 

    """

def pathjoin(*kwargs):
    """Joint strings to a path.

    Parameters
    ----------
    *kwargs
        strings.

    Returns
    -------
    str
        The joined path string.
    """

def fileparts(file):
    r"""Filename parts

    Returns the path, file name, and file name extension for the specified :attr:`file`.
    The :attr:`file` input is the name of a file or folder, and can include a path and
    file name extension.

    Parameters
    ----------
    file : str
        The name string of a file or folder.

    Returns
    -------
    filepath : str
        The path string of the file or folder.
    name : str
        The name string of the file or folder.
    ext : str
        The extension string of the file.

    """

def readtxt(filepath, mode=None):
    """Read a text file.

    Parameters
    ----------
    filepath : str
        The path string of the file.
    mode : str or None, optional
        ``'line'`` --> reading line-by-line.
        ``'all'`` --> reading all content.

    Returns
    -------
    str
        File content.
    """

def readnum(filepath, pmain='Train', psub='loss', vfn=float, nshots=None):
    """Read a file and extract numbers in it.

    Parameters
    ----------
    filepath : str
        Description
    pmain : str, optional
        The matching pattern string, such as '--->Train'.
    psub : str, optional
        The sub-matching pattern string, such as 'loss'.
    vfn : function, optional
        The function for formating the numbers. ``float`` --> convert to float number; ``int`` --> convert to integer number..., The default is ``float``.
    nshots : None, optional
        The number of shots of sub-matching pattern.

    Returns
    -------
    str
        The list of numbers.
    """

def readcsv(filepath, sep=None, vfn=None, nlines=None):
    """Read a csv file and extract numbers in it.

    Parameters
    ----------
    filepath : str
        The path string of the file.
    sep : str, optional
        The separation character. Such as ``','`` or ``' '``. If None (default) or ``''`` (empty) return a list of all the lines.
    vfn : function or None, optional
        The function for formating the numbers. ``float`` --> convert to float number; ``int`` --> convert to integer number..., The default is :obj:`None`, which means won't converted, string format.
    nlines : None, optional
        The number of lines for reading, the default is None, which means all the lines.

    Returns
    -------
    list
        The list of numbers or strings.
    """

def readsec(filepath, pmain='Train', psub='time: ', vfn=int, nshots=None):
    """Read a file and extract seconds in it.

        ``hh:mm:ss``  -->  ``hh*3600 + mm*60 + ss``

    Parameters
    ----------
    filepath : str
        The path string of the file.
    pmain : str, optional
        The matching pattern string, such as '--->Train'.
    psub : str, optional
        The sub-matching pattern string, such as 'loss'.
    vfn : function or None, optional
        The function for formating the numbers. ``float`` --> convert to float number; ``int`` --> convert to integer number.
    nshots : None, optional
        The number of shots of sub-matching pattern.

    Returns
    -------
    list
        The list of seconds.
    """

def fopen(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True):
    """file open

    difference to python's builtin function :func:`open` is when mode is ``'w'``, if the file is not empty,
    this function gives a selection of overwrite or skip.
    
    Parameters
    ----------
    file : str
        file path string.
    mode : str, optional
        ``'r'`` (read), ``'w'`` (overwrite), ``'a'`` (append), ..., see also :func:`open`, by default "r".
    buffering : int, optional
        see also :func:`open`, , by default -1
    encoding : str, optional
        see also :func:`open`, , by default None
    errors : str, optional
        see also :func:`open`, , by default None
    newline : str, optional
        see also :func:`open`, , by default None
    closefd : bool, optional
        see also :func:`open`, , by default True

    Returns
    -------
    TextIOWrapper or None
        
    """


