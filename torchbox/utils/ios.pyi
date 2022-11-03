def loadyaml(filepath, field=None):
    r"""Load a yaml file.

    Parameters
    ----------
    filepath : str
        The file path string.
    field : None, optional
        The string of field that want to be loaded.

    """

def saveyaml(filepath, ddict=None, indent='-', mode='w'):
    r"""Load a yaml file.

    Parameters
    ----------
    filepath : str
        The file path string.
    ddict : dict
        The data to be written is in dict format, {'field1': value1, ...}
    indent : str
        The indent, (the default is ``'  '``)
    mode : str
        save mode, ``'w'`` for overwrite, ``'a'`` for add.

    Returns
    -------
    0
        all is ok!

    """

def loadjson(filepath, field=None):
    """load a json file

    Parameters
    ----------
    filepath : str
        The file path string.
    field : None, optional
        The string of field that want to be loaded.

    """

def _check_keys(d):
    ...

def _todict(matobj):
    ...

def loadmat(filepath):
    """load data from an ``.mat`` file

    load data from an ``.mat`` file (``'None'`` will be replaced by :obj:`None`)

    see https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

    Parameters
    ----------
    filepath : str
        The file path string.

    """

def savemat(filepath, mdict, fmt='5'):
    """save data to an ``.mat`` file

    save data to ``.mat`` file (:obj:`None` will be replaced by ``'None'``)

    Parameters
    ----------
    filepath : str
        savefile path
    mdict : dict
        data in dict formation. 
    fmt : str, optional
        mat formation, by default '5'

    Returns
    -------
    0
        all is ok!
    """

def _create_group_dataset(group, mdict):
    ...

def _read_group_dataset(group, mdict, keys=None):
    ...

def loadh5(filename, keys=None):
    """load h5 file

    load data from a h5 file. (``'None'`` will be replaced by :obj:`None`)

    Parameters
    ----------
    filename : str
        File's full path string.
    keys : list
        list of keys.

    Returns
    -------
    D : dict
        The loaded data in ``dict`` type.

    """

def saveh5(filename, mdict, mode='w'):
    """save data to h5 file

    save data to h5 file (:obj:`None` will be replaced by ``'None'``)

    Parameters
    ----------
    filename : str
        filename string
    mdict : dict
        each dict is store in group, the elements in dict are store in dataset
    mode : str
        save mode, ``'w'`` for write, ``'a'`` for add.

    Returns
    -------
    number
        0 --> all is well.
    """

def mvkeyh5(filepath, ksf, kst, sep='.'):
    """rename keys in ``.h5`` file

    Parameters
    ----------
    filepath : str
        The file path string
    ksf : list
        keys from list, e.g. ['a.x', 'b.y']
    kst : list
        keys to list, e.g. ['a.1', 'b.2']
    sep : str, optional
        The separate pattern, default is ``'.'``

    Returns
    -------
    0
        All is ok!
    """

def loadbin(file, dbsize=4, dtype='i', endian='little', offsets=0, nshots=None):
    r"""load binary file

    load data from binary file

    Parameters
    ----------
    file : str
        the binary file path
    dbsize : int, optional
        number pf bits of each number, by default 4
    dtype : str, optional
        the type of data. 
        - ``'c'``:char, ``'b'``:schar, ``'B'``:uchar, ``'s'``:char[], ``'p'``:char[], ``'P'``:void* 
        - ``'h'``:short, ``'H'``:ushort, ``'i'``:int, ``'I'``:uint, ``'l'``:long, ``'L'``:ulong, ``'q'``:longlong, ``'Q'``:ulonglong
        - ``'f'``:float, ``'d'``:double, by default ``'i'``
    endian : str, optional
        byte order ``'little'`` / ``'l'``, ``'big'`` / ``'b'``, by default 'little'
    offsets : int, optional
        start reading offsets index, by default 0
    nshots : int, optional
        the number of data points to be read.

    Returns
    -------
    list
        loaded data.

    Examples
    ----------

    ::

        import torchbox as pb

        datafile = 'data/data.bin'

        x = [1, 3, 6, 111]
        pb.savebin('./data.bin', x, dtype='i', endian='L', mode='o')

        y = pb.loadbin('./data.bin', dbsize=4, dtype='i', endian='L')

        print(y)

        x = (1.3, 3.6)
        pb.savebin('./data.bin', x, dtype='f', endian='B', offsets=16, mode='a')

        y = pb.loadbin('./data.bin', dbsize=4, dtype='f', endian='B', offsets=16)

        print(y)
    
    """

def savebin(file, x, dtype='i', endian='little', offsets=0, mode='o'):
    r"""load binary file

    load data from binary file

    Parameters
    ----------
    file : str
        the binary file path
    x : any
        data to be written (iterable)
    dtype : str, optional
        the type of data. 
        - ``'c'``:char, ``'b'``:schar, ``'B'``:uchar, ``'s'``:char[], ``'p'``:char[], ``'P'``:void* 
        - ``'h'``:short, ``'H'``:ushort, ``'i'``:int, ``'I'``:uint, ``'l'``:long, ``'L'``:ulong, ``'q'``:longlong, ``'Q'``:ulonglong
        - ``'f'``:float, ``'d'``:double, by default ``'i'``
    endian : str, optional
        byte order ``'little'`` / ``'l'``, ``'big'`` / ``'b'``, by default 'little'
    offsets : int, optional
        start reading offsets index, by default 0
    mode : int, optional
        - ``'append'`` / ``'a'`` --> append data to the end of the file
        - ``'overwrite'`` / ``'o'`` --> overwrite the file. (default)

    Examples
    ----------

    ::

        import torchbox as pb

        datafile = 'data/data.bin'

        x = [1, 3, 6, 111]
        pb.savebin('./data.bin', x, dtype='i', endian='L', mode='o')

        y = pb.loadbin('./data.bin', dbsize=4, dtype='i', endian='L')

        print(y)

        x = (1.3, 3.6)
        pb.savebin('./data.bin', x, dtype='f', endian='B', offsets=16, mode='a')

        y = pb.loadbin('./data.bin', dbsize=4, dtype='f', endian='B', offsets=16)

        print(y)

    """


