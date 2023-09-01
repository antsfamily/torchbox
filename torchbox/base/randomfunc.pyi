def setseed(seed=None, target='torch'):
    r"""set seed

    Set numpy / random / torch / torch.random / torch.cuda seed.

    Parameters
    ----------
    seed : int or None, optional
        seed for random number generator (the default is None)
    target : str, optional
        - ``'numpy'``: ``np.random.seed(seed)``
        - ``'random'``: ``random.seed(seed)``
        - ``'torch'``: ``torch.manual_seed(seed)`` (default)
        - ``'torch.random'``: ``torch.random.manual_seed(seed)``
        - ``'cuda'``: ``torch.cuda.manual_seed(seed)``
        - ``'cudaall'``: ``torch.cuda.manual_seed_all(seed)``

    """

def permutation(x):
    r"""permutation function like numpy.random.permutation

    permutation function like numpy.random.permutation

    Parameters
    ----------
    x : Tensor
        inputs, can have any dimensions.

    Returns
    -------
    x : Tensor
        permutated tensor

    """

def randperm(start, stop, n):
    r"""randperm function like matlab

    genarates diffrent random interges in range [start, stop)

    Parameters
    ----------
    start : int or list
        start sampling point
    stop : int or list
        stop sampling point
    n : int, list or None
        the number of samples (default None, int((stop - start)))

    see also :func:`randgrid`.
    """

def randperm2d(H, W, number, population=None, mask=None):
    r"""randperm 2d function

    genarates diffrent random interges in range [start, end)

    Parameters
    ----------
    H : int
        height
    W : int
        width
    number : int
        random numbers
    population : {list or numpy array(1d or 2d)}
        part of population in range(0, H*W)

    """

def randgrid(start, stop, step, shake=0, n=None):
    r"""generates non-repeated uniform stepped random ints

    Generates :attr:`n` non-repeated random ints from :attr:`start` to :attr:`stop`
    with step size :attr:`step`.

    When step is 1 and shake is 0, it works similar to randperm,

    Parameters
    ----------
    start : int or list
        start sampling point
    stop : int or list
        stop sampling point
    step : int or list
        sampling stepsize
    shake : float
        the shake rate, if :attr:`shake` is 0, no shake, (default),
        if positive, add a positive shake, if negative, add a negative.
    n : int or None
        the number of samples (default None, int((stop0 - start0) / step0) * int((stop1 - start1) / step1)...).

    Returns
    -------
        for multi-dimension, return a 2-d tensor, for 1-dimension, return a 1d-tensor.

    Example
    -------

    ::

        import matplotlib.pyplot as plt

        setseed(2021)
        print(randperm(2, 40, 8), ", randperm(2, 40, 8)")
        print(randgrid(2, 40, 1, -1., 8), ", randgrid(2, 40, 1, 8, -1.)")
        print(randgrid(2, 40, 6, -1, 8), ", randgrid(2, 40, 6, 8)")
        print(randgrid(2, 40, 6, 0.5, 8), ", randgrid(2, 40, 6, 8, 0.5)")
        print(randgrid(2, 40, 6, -1, 12), ", randgrid(2, 40, 6, 12)")
        print(randgrid(2, 40, 6, 0.5, 12), ", randgrid(2, 40, 6, 12, 0.5)")

        mask = th.zeros((5, 6))
        mask[3, 4] = 0
        mask[2, 5] = 0

        Rh, Rw = randperm2d(5, 6, 4, mask=mask)

        print(Rh)
        print(Rw)

        y = randperm(0, 8192, 800)
        x = randperm(0, 8192, 800)

        y, x = randgrid([0, 0], [512, 512], [64, 64], [0.0, 0.], 32)
        print(len(y), len(x))

        plt.figure()
        plt.plot(x, y, 'o')
        plt.show()

        y, x = randgrid([0, 0], [8192, 8192], [256, 256], [0., 0.], 400)
        print(len(y), len(x))

        plt.figure()
        plt.plot(x, y, '*')
        plt.show()


    see also :func:`randperm`.

    """


