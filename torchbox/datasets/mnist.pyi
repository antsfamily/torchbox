def read_mnist(rootdir, dataset='test', fmt='bin'):
    """read mnist dataset

    The data can be downloaded from http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    rootdir : str
        root directory path string of mnist dataset 
    dataset : str, optional
        dataset to be read, ``'test'`` or ``'train'``, by default 'test'.
    fmt : str, optional
        the dataset formation, ``'bin'`` (original) or ``'img'`` (image), by default 'bin'.

    Returns
    -------
    list or tuple
        X : tensor
            image data
        Y : tensor
            label data
    
    Examples
    --------

        ::

            rootdir = '/mnt/d/DataSets/oi/dgi/mnist/pics/'
            dataset = 'test'
            X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='img')
            print(X.shape, Y.shape)

            rootdir = '/mnt/d/DataSets/oi/dgi/mnist/lecun/'
            dataset = 'train'
            X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='bin')
            print(X.shape, Y.shape)
            dataset = 'test'
            X, Y = read_mnist(rootdir=rootdir, dataset=dataset, fmt='bin')
            print(X.shape, Y.shape)

            # output
            (10000, 28, 28) (10000,)
            (60000, 28, 28) (60000,)
            (10000, 28, 28) (10000,)
    """


