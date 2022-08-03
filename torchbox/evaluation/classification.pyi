def accuracy(X, Y, TH=None):
    r"""compute accuracy



    Parameters
    ----------
    X : tensor
        Predicted one hot matrix, :math:`\{0, 1\}`
    Y : tensor
        Referenced one hot matrix, :math:`\{0, 1\}`
    TH : float, optional
        threshold: X > TH --> 1, X <= TH --> 0
    """


