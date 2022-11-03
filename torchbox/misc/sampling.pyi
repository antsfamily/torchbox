def slidegrid(start, stop, step, shake=0, n=None):
    r"""generates sliding grid indexes

    Generates :attr:`n` sliding grid indexes from :attr:`start` to :attr:`stop`
    with step size :attr:`step`.

    Args:
        start (int or list): start sampling point
        stop (int or list): stop sampling point
        step (int or list): sampling stepsize
        shake (float): the shake rate, if :attr:`shake` is 0, no shake, (default),
            if positive, add a positive shake, if negative, add a negative.
        n (int or None): the number of samples (default None, int((stop0 - start0) / step0) * int((stop1 - start1) / step1)...).

    Returns:
        for multi-dimension, return a 2-d tensor, for 1-dimension, return a 1d-tensor.

    Raises:
        TypeError: The number of samples should be an integer or None.

    see :func:`randperm`, :func:`randgrid`.

    """

def dnsampling(x, ratio=1., axis=-1, smode='uniform', omode='discard', seed=None, extra=False):
    """Summary

    Args:
        x (Tensor): The Input tensor.
        ratio (float, optional): Downsampling ratio.
        axis (int, optional): Downsampling axis (default -1).
        smode (str, optional): Downsampling mode: ``'uniform'``, ``'random'``, ``'random2'``.
        omode (str, optional): output mode: ``'discard'`` for discarding, ``'zero'`` for zero filling.
        seed (int or None, optional): seed for torch's random.
        extra (bool, optional): If ``True``, also return sampling mask.

    Returns:
        (Tensor): Description

    Raises:
        TypeError: :attr:`axis`
        ValueError: :attr:`ratio`, attr:`smode`, attr:`omode`
    """

def sample_tensor(x, n, axis=0, groups=1, mode='sequentially', seed=None, extra=False):
    """sample a tensor

    Sample a tensor sequentially/uniformly/randomly.

    Args:
        x (torch.Tensor): a torch tensor to be sampled
        n (int): sample number
        axis (int, optional): the axis to be sampled (the default is 0)
        groups (int, optional): number of groups in this tensor (the default is 1)
        mode (str, optional): - ``'sequentially'``: evenly spaced (default)
            - ``'uniformly'``: [0, int(n/groups)]
            - ``'randomly'``: randomly selected, non-returned sampling
        seed (None or int, optional): only work for ``'randomly'`` mode (the default is None)
        extra (bool, optional): If ``True``, also return the selected indexes, the default is ``False``.

    Returns:
        y (torch.Tensor): Sampled torch tensor.
        idx (list): Sampled indexes, if :attr:`extra` is ``True``, this will also be returned.


    Example:
    
        ::

            setseed(2020, 'torch')

            x = th.randint(1000, (20, 3, 4))
            y1, idx1 = sample_tensor(x, 10, axis=0, groups=2, mode='sequentially', extra=True)
            y2, idx2 = sample_tensor(x, 10, axis=0, groups=2, mode='uniformly', extra=True)
            y3, idx3 = sample_tensor(x, 10, axis=0, groups=2, mode='randomly', extra=True)

            print(x.shape)
            print(y1.shape)
            print(y2.shape)
            print(y3.shape)
            print(idx1)
            print(idx2)
            print(idx3)

            the outputs are as follows:

            torch.Size([20, 3, 4])
            torch.Size([10, 3, 4])
            torch.Size([10, 3, 4])
            torch.Size([10, 3, 4])
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
            [3, 1, 5, 8, 7, 17, 18, 13, 16, 10]


    Raises:
        ValueError: The tensor does not has enough samples.


    """

def shuffle_tensor(x, axis=0, groups=1, mode='inter', seed=None, extra=False):
    """shuffle a tensor

    Shuffle a tensor randomly.

    Args:
        x (Tensor): A torch tensor to be shuffled.
        axis (int, optional): The axis to be shuffled (default 0)
        groups (number, optional): The number of groups in this tensor (default 1)
        mode (str, optional):
            - ``'inter'``: between groups (default)
            - ``'intra'``: within group
            - ``'whole'``: the whole
        seed (None or number, optional): random seed (the default is None)
        extra (bool, optional): If ``True``, also returns the shuffle indexes, the default is ``False``.

    Returns:
        y (Tensor): Shuffled torch tensor.
        idx (list): Shuffled indexes, if :attr:`extra` is ``True``, this will also be returned.


    Example:

        ::

            setseed(2020, 'torch')

            x = th.randint(1000, (20, 3, 4))
            y1, idx1 = shuffle_tensor(x, axis=0, groups=4, mode='intra', extra=True)
            y2, idx2 = shuffle_tensor(x, axis=0, groups=4, mode='inter', extra=True)
            y3, idx3 = shuffle_tensor(x, axis=0, groups=4, mode='whole', extra=True)

            print(x.shape)
            print(y1.shape)
            print(y2.shape)
            print(y3.shape)
            print(idx1)
            print(idx2)
            print(idx3)

            the outputs are as follows:

            torch.Size([20, 3, 4])
            torch.Size([20, 3, 4])
            torch.Size([20, 3, 4])
            torch.Size([20, 3, 4])
            [1, 0, 3, 4, 2, 8, 6, 5, 9, 7, 13, 11, 12, 14, 10, 18, 15, 17, 16, 19]
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
            [1, 13, 12, 5, 19, 9, 11, 6, 4, 16, 17, 3, 8, 18, 7, 10, 15, 0, 14, 2]


    """

def split_tensor(x, ratios=[0.7, 0.2, 0.1], axis=0, shuffle=False, seed=None, extra=False):
    """split tensor

    split a tensor into some parts.

    Args:
        x (Tensor): A torch tensor.
        ratios (list, optional): Split ratios (the default is [0.7, 0.2, 0.05])
        axis (int, optional): Split axis (the default is 0)
        shuffle (bool, optional): Whether shuffle (the default is False)
        seed (int, optional): Shuffule seed (the default is None)
        extra (bool, optional): If ``True``, also return the split indexes, the default is ``False``.

    Returns:
        (list of Tensor): Splitted tensors.
    """

def tensor2patch(x, n=None, size=(32, 32), axis=(0, 1), start=(0, 0), stop=(None, None), step=(1, 1), shake=(0, 0), mode='slidegrid', seed=None):
    """sample patch from a tensor

    Sample some patches from a tensor, tensor and patch can be any size.

    Args:
        x (Tensor): Tensor to be sampled.
        n (int, optional): The number of pactches, the default is None, auto computed,
            equals to the number of blocks with specified :attr:`step`
        size (tuple or int, optional): The size of patch (the default is (32, 32))
        axis (tuple or int, optional): The sampling axis (the default is (0, 1))
        start (tuple or int, optional): Start sampling index for each axis (the default is (0, 0))
        stop (tuple or int, optional): Stopp sampling index for each axis. (the default is (None, None), which [default_description])
        step (tuple or int, optional): Sampling stepsize for each axis  (the default is (1, 1), which [default_description])
        shake (tuple or int or float, optional): float for shake rate, int for shake points (the default is (0, 0), which means no shake)
        mode (str, optional): Sampling mode, ``'slidegrid'``, ``'randgrid'``, ``'randperm'`` (the default is 'slidegrid')
        seed (int, optional): Random seed. (the default is None, which means no seed.)

    Returns:
        (Tensor): A Tensor of sampled patches.
    """

def patch2tensor(p, size=(256, 256), axis=(1, 2), start=(0, 0), stop=(None, None), step=None, mode='nfirst'):
    """merge patch to a tensor


    Args:
        p (Tensor): A tensor of patches.
        size (tuple, optional): Merged tensor size in the dimension (the default is (256, 256)).
        axis (tuple, optional): Merged axis of patch (the default is (1, 2))
        start (tuple, optional): start position for placing patch (the default is (0, 0))
        stop (tuple, optional): stop position for placing patch (the default is (0, 0))
        step (tuple, optional): step size for placing patch (the default is ``'None'``, which means the size of patch)
        mode (str, optional): Patch mode ``'nfirst'`` or ``'nlast'`` (the default is 'nfirst',
            which means the first dimension is the number of patches)

    Returns:
        Tensor: Merged tensor.
    """

def read_samples(datafiles, keys=[['SI', 'ca', 'cr']], nsamples=[10], groups=[1], mode='sequentially', axis=0, parts=None, seed=None):
    """Read samples

    Args:
        datafiles (list): list of path strings
        keys (list, optional): data keys to be read
        nsamples (list, optional): number of samples for each data file
        groups (list, optional): number of groups in each data file
        mode (str, optional): sampling mode for all datafiles
        axis (int, optional): sampling axis for all datafiles
        parts (None, optional): number of parts (split samples into some parts)
        seed (None, optional): the seed for random stream

    Returns:
        tensor: samples

    Raises:
        ValueError: :attr:`nsamples` should be large enough
    """


