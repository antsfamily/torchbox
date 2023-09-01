def categorical2onehot(X, nclass=None):
    r"""converts categorical to onehot

    Parameters
    ----------
    X : list or tensor
        the categorical list or tensor
    nclass : int, optional
        the number of classes, by default None (auto detected)

    Returns
    -------
    tensor
        the one-hot matrix
    """    

def onehot2categorical(X, axis=-1, offset=0):
    r"""converts onehot to categorical

    Parameters
    ----------
    X : list or tensor
        the one-hot tensor
    axis : int, optional
        the axis for one-hot encoding, by default -1
    offset : int, optional
        category label head, by default 0

    Returns
    -------
    tensor
        the category label
    """    

def accuracy(P, T, axis=None):
    r"""computes the accuracy

    .. math::
       A = \frac{\sum(P==T)}{N}

    where :math:`N` is the number of samples.

    Parameters
    ----------
    P : list or tensor
        predicted label (categorical or one-hot)
    T : list or tensor
        target label (categorical or one-hot)
    axis : int, optional
        the one-hot encoding axis, by default None, which means :attr:`P` and :attr:`T` are categorical.

    Returns
    -------
    float
        the accuracy

    Raises
    ------
    ValueError
        :attr:`P` and :attr:`T` should have the same shape!
    ValueError
        You should specify the one-hot encoding axis when :attr:`P` and :attr:`T` are in one-hot formation!

    Examples
    --------

    ::

        import torchbox as tb

        T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
        P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

        print(tb.accuracy(P, T))

        #---output
        0.8 

    """

def confusion(P, T, axis=None, cmpmode='...'):
    r"""computes the confusion matrix

    Parameters
    ----------
    P : list or tensor
        predicted label (categorical or one-hot)
    T : list or tensor
        target label (categorical or one-hot)
    axis : int, optional
        the one-hot encoding axis, by default None, which means :attr:`P` and :attr:`T` are categorical.
    cmpmode : str, optional
        ``'...'`` for one-by one mode, ``'@'`` for multiplication mode (:math:`P^TT`), by default '...'

    Returns
    -------
    tensor
        the confusion matrix

    Raises
    ------
    ValueError
        :attr:`P` and :attr:`T` should have the same shape!
    ValueError
        You should specify the one-hot encoding axis when :attr:`P` and :attr:`T` are in one-hot formation!

    Examples
    --------

    ::

        import torchbox as tb

        T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
        P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

        C = tb.confusion(P, T, cmpmode='...')
        print(C)
        C = tb.confusion(P, T, cmpmode='@')
        print(C)

        #---output
        [[3. 0. 2. 0. 1. 0.]
        [0. 3. 0. 0. 0. 0.]
        [1. 0. 1. 0. 0. 0.]
        [0. 0. 0. 3. 0. 0.]
        [0. 0. 0. 0. 3. 0.]
        [0. 0. 0. 0. 0. 3.]]
        [[3. 0. 2. 0. 1. 0.]
        [0. 3. 0. 0. 0. 0.]
        [1. 0. 1. 0. 0. 0.]
        [0. 0. 0. 3. 0. 0.]
        [0. 0. 0. 0. 3. 0.]
        [0. 0. 0. 0. 0. 3.]]

    """    

def kappa(C):
    r"""computes kappa

    .. math::
       K = \frac{p_o - p_e}{1 - p_e}

    where :math:`p_o` and :math:`p_e` can be obtained by

    .. math::
        p_o = \frac{\sum_iC_{ii}}{\sum_i\sum_jC_{ij}} 
    
    .. math::
        p_e = \frac{\sum_j\left(\sum_iC_{ij}\sum_iC_{ji}\right)}{\sum_i\sum_jC_{ij}} 

    Parameters
    ----------
    C : Tensor
        The confusion matrix

    Returns
    -------
    float
        The kappa value.

    Examples
    --------

    ::

        import torchbox as tb

        T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
        P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

        C = tb.confusion(P, T, cmpmode='...')
        print(tb.kappa(C))
        print(tb.kappa(C.T))

        #---output
        0.7583081570996979
        0.7583081570996979

    """

def plot_confusion(C, cmap=None, mode='rich', xticks='label', yticks='label', xlabel='Target', ylabel='Predicted', title='Confusion matrix', **kwargs):
    r"""plots confusion matrix.

    plots confusion matrix.

    Parameters
    ----------
    C : Tensor
        The confusion matrix
    cmap : None or str, optional
        The colormap, by default :obj:`None`, which means our default configuration (green-coral)
    mode : str, optional
        ``'pure'``, ``'bare'``, ``'simple'`` or ``'rich'``
    xticks : str, tuple or list, optional
        ``'label'`` --> class labels, or you can specify class name list, by default ``'label'``
    yticks : str, tuple or list, optional
        ``'label'`` --> class labels, or you can specify class name list, by default ``'label'``
    xlabel : str, optional
        The label string of axis-x, by default 'Target'
    ylabel : str, optional
        The label string of axis-y, by default 'Predicted'
    title : str, optional
        The title string, by default 'Confusion matrix'
    kwargs :
        linespacing : float
            The line spacing of text, by default ``0.15``
        numftd : dict
            The font dict of integer value, by default ::

                dict(fontsize=12, color='black', 
                     family='Times New Roman', 
                     weight='bold', style='normal')
        pctftd : dict
            The font dict of percent value, by default ::
            
                dict(fontsize=12, color='black', 
                     family='Times New Roman', 
                     weight='light', style='normal')
        restftd : dict
            The font dict of label, title and ticks, by default ::

                dict(fontsize=12, color='black', 
                     family='Times New Roman', 
                     weight='light', style='normal')
        pctfmt : str
            the format of percent value, such as ``'%.xf'`` means formating with two decimal places, by default ``'%.1f'``

    Returns
    -------
    pyplot
        pyplot handle

    Example
    -------

    .. image:: ./_static/ConfusionMatrixSimple.png
       :scale: 100 %
       :align: center

    .. image:: ./_static/ConfusionMatrixRich.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torchbox as tb

        T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
        P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

        C = tb.confusion(P, T, cmpmode='@')

        plt = tb.plot_confusion(C, cmap=None, mode='simple')
        plt = tb.plot_confusion(C, cmap='summer', mode='simple')
        plt.show()

        plt = tb.plot_confusion(C, cmap=None, mode='rich')
        plt = tb.plot_confusion(C, cmap='summer', mode='rich')
        plt.show()

    """    


