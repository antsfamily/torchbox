def cplot(ca, lmod=None):
def plots(x, ydict, plotdir='./', xlabel='x', ylabel='y', title='', issave=False, isshow=True):
class Plots:
    def __init__(self, plotdir='./', xlabel='x', ylabel='y', title='', figname=None, issave=False, isshow=True):
    def __call__(self, x, ydict, figname=None):
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
        see :func:`matplotlib.pyplot.imshow`

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
        Ys : list or tuple
        
        for others, see :func:`matplotlib.pyplot.plot_surface`

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
        
        for others, see :func:`matplotlib.pyplot.plot_surface`

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


    """

