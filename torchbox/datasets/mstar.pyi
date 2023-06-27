def mstar_header(filepath):
    r"""read header information of mstar file

    Parameters
    ----------
    filepath : str
        the mstar file path string.

    Returns
    -------
    dict
        header information dictionary.

    Examples
    --------

    The following example shows how to read the header information.

    ::

        import torchbox as tb

        datapath = tb.data_path('mstar')
        filepath = datapath + 'BTR70_HB03787.004'

        header = tb.mstar_header(filepath)
        for k, v in header.items():
            print(k, v)

    """

def mstar_raw(filepath, ofmt='c'):
    r"""load mstar raw data

    Each file is constructed with a prepended, variable-length, 
    Phoenix formatted (ASCII) header which contains detailed ground 
    truth and sensor information for the specific chip.  Following 
    the Phoenix header is the data block.  The data block is written 
    in Sun floating point format and is divided into two blocks, a 
    magnitude block followed by a phase block.  Byte swapping may be 
    required for certain host platforms.  Tools for reading and 
    manipulating the header information may be found at 
    https://www.sdms.afrl.af.mil .

    Parameters
    ----------
    filepath : str
        the data file path string.
    ofmt : str, optional
        output data type formation, ``'ap'`` for amplitude and angle,
        ``'c'`` for complex, and ``'r'`` for real and imaginary.

    Returns
    -------
    tensor
        the raw data with size :math:`{\mathbb C}^{H\times W}` (``'c'``), 
        :math:`{\mathbb R}^{H\times W \times 2}` (``'r'`` or ``'ap'``)

    Examples
    --------

    Read mstar raw amplitude-phase data and show in a figure.

    .. image:: ./_static/SHOW1_BTR70_HB03787.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torchbox as tb
        import matplotlib.pyplot as plt

        filepath = datapath + 'BTR70_HB03787.004'
        x = tb.mstar_raw(filepath, ofmt='ap')
        print(x.shape, th.max(x), th.min(x))

        plt.figure()
        plt.subplot(121)
        plt.imshow(x[..., 0])
        plt.title('amplitude')
        plt.subplot(122)
        plt.imshow(x[..., 1])
        plt.title('phase')
        plt.show()


    Read mstar raw complex-valued data and show in a figure.

    .. image:: ./_static/SHOW_BTR70_HB03787.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torchbox as tb
        import matplotlib.pyplot as plt

        filepath = datapath + 'BTR70_HB03787.004'
        x = tb.mstar_raw(filepath, ofmt='c')
        print(x.shape, th.max(x.abs()), th.min(x.abs()))

        plt.figure()
        plt.subplot(221)
        plt.imshow(x.real)
        plt.title('real part')
        plt.subplot(222)
        plt.imshow(x.imag)
        plt.title('imaginary part')
        plt.subplot(223)
        plt.imshow(x.abs())
        plt.title('amplitude')
        plt.subplot(224)
        plt.imshow(x.angle())
        plt.title('phase')
        plt.show()

    """

# def read_mstar(rootdir, dataset='test', fmt='bin'):
      ...


