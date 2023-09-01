def cplot(ca, lmod=None):
    ...

def plots(x, ydict, plotdir='./', xlabel='x', ylabel='y', title='', issave=False, isshow=True):
    ...

class Plots:
    ...

    def __init__(self, plotdir='./', xlabel='x', ylabel='y', title='', figname=None, issave=False, isshow=True):
        ...

    def __call__(self, x, ydict, figname=None):
        ...

def plot(Ys, nrows=None, ncols=None, styles=None, legends=None, grids=False, xlabels=None, ylabels=None, titles=None, figsize=None, outfile=None, **kwargs):
    r"""show images

    This function create an figure and show images in :math:`a` rows and :math:`b` columns.

    Parameters
    ----------
    Ys : array, list or tuple
        list/tuple of image arrays, if the type is not list or tuple, wrap it.
    nrows : int, optional
        show in :attr:`nrows` rows, by default None (auto computed).
    ncols : int, optional
        show in :attr:`ncols` columns, by default None (auto computed).
    styles : str or list, optional
        line style
    legends : str or list, optional
        legend str list
    grids : bool, optional
        If :obj:`True` plot grid, default :obj:`False`.
    xlabels : str, optional
        labels of x-axis
    ylabels : str, optional
        labels of y-axis
    titles : str, optional
        titles
    figsize : tuple, optional
        figure size, by default None
    outfile : str, optional
        save image to file, by default None (do not save).
    kwargs : 
        fig : figure handle
            sunch as ``fig = plt.figure()``
        Xs : list or tuple
            Y-axis values

        see :func:`matplotlib.pyplot.plot`

    see also :func:`~torchbox.utils.plot_show.imshow`, :func:`~torchbox.utils.plot_show.mesh`, :func:`~torchbox.utils.plot_show.mshow`.

    Returns
    -------
    plt
        plot handle

    Examples
    ---------

    ::

        x1 = np.random.rand(2, 100)
        x2 = np.random.rand(2, 100)
        plt = plot([[xi for xi in x1], [xi for xi in x2]])
        plt.show()

        x1 = np.random.rand(2, 100)
        x2 = np.random.rand(2, 100)
        plt = plot([[xi for xi in x1], [xi for xi in x2]], styles=[['-b', '-r'], ['-b', '-r']], legends=[['real', 'imag'], ['real', 'imag']], grids=True)
        plt.show()

    """

def imshow(Xs, nrows=None, ncols=None, xlabels=None, ylabels=None, titles=None, figsize=None, outfile=None, **kwargs):
    r"""show images

    This function create an figure and show images in :math:`a` rows and :math:`b` columns.

    Parameters
    ----------
    Xs : array, list or tuple
        list/tuple of image arrays, if the type is not list or tuple, wrap it.
    nrows : int, optional
        show in :attr:`nrows` rows, by default None (auto computed).
    ncols : int, optional
        show in :attr:`ncols` columns, by default None (auto computed).
    xlabels : str, optional
        labels of x-axis
    ylabels : str, optional
        labels of y-axis
    titles : str, optional
        titles
    figsize : tuple, optional
        figure size, by default None
    outfile : str, optional
        save image to file, by default None (do not save).
    kwargs : 
        fig : figure handle
            sunch as ``fig = plt.figure()``

        see also :func:`matplotlib.pyplot.imshow`

    see also :func:`~torchbox.utils.plot_show.plot`, :func:`~torchbox.utils.plot_show.mesh`, :func:`~torchbox.utils.plot_show.mshow`.
        
    Returns
    -------
    plt
        plot handle

    Examples
    ---------

    ::

        x = np.random.rand(3, 100, 100)
        plt = imshow([xi for xi in x])
        plt.show()

        # ---animation
        x = np.random.rand(10, 128, 128)
        y = np.random.rand(10, 128, 128)
        fig = plt.figure()
        for n in range(10):
            fig.clf()
            plt = imshow([x[n], y[n]], 1, 2, fig=fig)
            plt.pause(0.5)

    """

def mesh(Zs, nrows=None, ncols=None, xlabels=None, ylabels=None, zlabels=None, titles=None, figsize=None, outfile=None, **kwargs):
    r"""mesh

    This function create an figure and show some 2d-arrays in :math:`a` rows and :math:`b` columns with 3d projection.

    Parameters
    ----------
    Zs : array, list or tuple
        list/tuple of image arrays, if the type is not list or tuple, wrap it.
    nrows : int, optional
        show in :attr:`nrows` rows, by default None (auto computed).
    ncols : int, optional
        show in :attr:`ncols` columns, by default None (auto computed).
    xlabels : str, optional
        labels of x-axis
    ylabels : str, optional
        labels of y-axis
    zlabels : str, optional
        labels of z-axis
    titles : str, optional
        titles
    figsize : tuple, optional
        figure size, by default None
    outfile : str, optional
        save image to file, by default None (do not save).
    kwargs : 
        Xs : list or tuple
            X-axis values
        Ys : list or tuple
            Y-axis values
        fig : figure handle
            sunch as ``fig = plt.figure()``
        
        for other kwargs, see also :func:`matplotlib.pyplot.plot_surface`

    see also :func:`~torchbox.utils.plot_show.imshow`, :func:`~torchbox.utils.plot_show.plot`, :func:`~torchbox.utils.plot_show.mshow`.
        
    Returns
    -------
    plt
        plot handle

    Examples
    ---------

    ::

        x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 20))
        z = np.random.rand(20, 10)

        plt = mesh(z, 1, 2)
        plt.show()
        
        plt = mesh(z, 1, 2, Xs=[np.arange(30, 40)])
        plt.show()

        # ---animation
        x = np.random.rand(10, 128, 128)
        y = np.random.rand(10, 128, 128)
        fig = plt.figure()
        for n in range(10):
            fig.clf()
            plt = mesh([x[n], y[n]], 1, 2, fig=fig)
            plt.pause(0.5)

    """

def mshow(Zs, nrows=None, ncols=None, xlabels=None, ylabels=None, zlabels=None, titles=None, projections=None, figsize=None, outfile=None, **kwargs):
    r"""show tensors

    This function create an figure and show some 2d-arrays in :math:`a` rows and :math:`b` columns with 2d/3d projection.

    Parameters
    ----------
    Zs : array, list or tuple
        list/tuple of image arrays, if the type is not list or tuple, wrap it.
    nrows : int, optional
        show in :attr:`nrows` rows, by default None (auto computed).
    ncols : int, optional
        show in :attr:`ncols` columns, by default None (auto computed).
    xlabels : str, optional
        labels of x-axis
    ylabels : str, optional
        labels of y-axis
    zlabels : str, optional
        labels of z-axis
    titles : str, optional
        titles
    figsize : tuple, optional
        figure size, by default None
    outfile : str, optional
        save image to file, by default None (do not save).
    kwargs : 
        Xs : list or tuple
        Ys : list or tuple
        fig : figure handle
            sunch as ``fig = plt.figure()``

        for others, see also :func:`matplotlib.pyplot.plot_surface`

    see also :func:`~torchbox.utils.plot_show.imshow`, :func:`~torchbox.utils.plot_show.mesh`, :func:`~torchbox.utils.plot_show.plot`.
        
    Returns
    -------
    plt
        plot handle

    Examples
    ---------

    ::

        x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 20))
        z1 = np.random.rand(20, 10)
        z2 = np.random.randn(60, 60)

        plt = mshow([z1, z2], 1, 2, Xs=[np.arange(30, 40)], projections=['3d', '2d'])
        plt.show()

        # ---animation
        x = np.random.rand(10, 128, 128)
        y = np.random.rand(10, 128, 128)
        fig = plt.figure()
        for n in range(10):
            fig.clf()
            plt = mshow([x[n], y[n]], 1, 2, fig=plt)
            plt.pause(0.5)

    """


