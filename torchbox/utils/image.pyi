def imread(imgfile):
    ...

def imsave(outfile, img):
    ...

def imadjust(img, lhin, lhout):
    ...

def imadjustlog(img, lhin=None, lhout=None):
    ...

def histeq(img, nbins=256):
    ...

def imresize(img, oshape=None, odtype=None, order=1, mode='constant', cval=0, clip=True, preserve_range=False):
    """resize image to oshape

    see :func:`skimage.transform.resize`.

    Parameters
    ----------
    img : tensor
        Input image.
    oshape : tuple, optional
        output shape (the default is None, which is the same as the input)
    odtype : str, optional
        output data type, ``'uint8', 'uint16', 'int8', ...`` (the default is None, float)
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode. {``'constant'``, ``'edge'``, ``'symmetric'``, ``'reflect'``, ``'wrap'``},
        Modes match the behaviour of `numpy.pad`.  The
        default mode is 'constant'.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.

    Returns
    -------
    resized : tensor
        Resized version of the input.

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    Examples
    --------
    ::

        >>> from skimage import data
        >>> from skimage.transform import resize
        >>> image = data.camera()
        >>> resize(image, (100, 100), mode='reflect').shape
        (100, 100)

    """


