def rgb2gray(rgb, fmt='chnllast'):
    r"""Converts RGB image to GRAY image

    Converts RGB image to GRAY image according to

    .. math::
       G = 0.2989R + 0.5870G + 0.1140B

    see matlab's ``rgb2gray`` function for details.

    see also :func:`~torchbox.utils.colors.gray2rgb`.
    
    
    Parameters
    ----------
    rgb : Tensor
        Original RGB tensor.
    fmt : str, optional
        Specifies the position of channels in :attr:`rgb` tensor, surpported are:
        - ``'chnllast'`` (default)
        - ``'chnlfirst'``
    """

def gray2rgb(gray, cmap, drange=[0, 255], fmtstr=False):
    r"""Converts gray images to rgb image

    converts `gray` images to rgb image according to the specified colormap string 'cmap',
    supported are: ``'parula'``, ``'jet'``, ``'hsv'``, ``'hot'``, ``'cool'``, ``'spring'``,
    ``'summer'``, ``'autumn'``, ``'winter'``, ``'gray'``, ``'bone'``, ``'copper'``, ``'pink'``,
    ``'lines'``, ``'colorcube'``, ``'prism'``, ``'flag'``, ``'white'``.

    Parameters
    ----------
    gray : torch tensor
        The gray image :math:`N\times H\times W`, or the gray images :math:`N\times H\times W`, :math:`N` is the number of images.
    cmap : str or colormap
        colormap string
    drange : list or tuple
        The low and high value, default is ``[0, 255]``
    fmtstr : bool or str
        Specifies the format (``'int'``, ``'uint8'``, ``'uint16'``, ``'uint32'``, ``'float16'``,
        ``'float32'``, , ``'float64'``) for the output rgb matrix. (default is ``False``, don't change)

    see also :func:`~torchbox.utils.colors.rgb2gray`.

    Examples
    --------

    ::

        import torchbox as tb
        import matplotlib.pyplot as plt
        import torch as th

        cmap = 'jet'
        # cmap = 'hsv'
        # cmap = 'hot'
        # cmap = 'parula'
        gray = ts.imread('../../data/images/LenaGRAY256.png')
        print(gray.shape)

        rgb = gray2rgb(gray, cmap, [0, 1], False)  # rgb --> double, [0, 1]
        # rgb = gray2rgb(gray, cmap, [0, 255], False)  # rgb --> double, [0., 255.]
        # rgb = gray2rgb(gray, cmap, [0, 255], True)  # rgb --> uint8, [0, 255]

        print(gray.shape, th.min(gray), th.max(gray), gray.dtype)
        print(rgb.shape, th.min(rgb), th.max(rgb), rgb.dtype)

        plt.figure()
        plt.subplot(121)
        plt.imshow(gray, cmap=pb.parula if cmap == 'parula' else cmap)
        plt.subplot(122)
        plt.imshow(rgb)
        plt.show()
    """


