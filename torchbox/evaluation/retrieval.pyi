def true_positive(X, Y):
    """Find true positive elements

    true_positive(X, Y) returns those elements that are positive classes in Y
    and retrieved as positive in X.

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0

    Returns
    -------
    TP: Tensor
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In TP, true positive elements are ones, while others are zeros.
    """

def false_positive(X, Y):
    """Find false positive elements

    false_positive(X, Y) returns elements that are negative classes in Y
    and retrieved as positive in X.

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0

    Returns
    -------
    FP: Tensor
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In FP, false positive elements are ones, while others are zeros.
    """

def true_negative(X, Y):
    """Find true negative elements

    true_negative(X, Y) returns elements that are negative classes in Y
    and retrieved as negative in X.

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0

    Returns
    -------
    TN: Tensor
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In TN, true negative elements are ones, while others are zeros.
    """

def false_negative(X, Y):
    """Find false negative elements

    true_negative(X, Y) returns elements that are positive classes in Y
    and retrieved as negative in X.

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0

    Returns
    -------
    FN: Tensor
        a torch tensor which has the same type with :attr:`X` or :attr:`Y`.
        In FN, false negative elements are ones, while others are zeros.
    """

def precision(X, Y, TH=None):
    r"""Compute precision

    .. math::
       {\rm PPV} = {P} = \frac{\rm TP}{{\rm TP} + {\rm FP}}
       :label: equ-Precision

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    P: float
        precision
    """

def recall(X, Y, TH=None):
    r"""Compute recall(sensitivity)

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    R: float
        recall
    """

def sensitivity(X, Y, TH=None):
    r"""Compute sensitivity(recall)

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    R: float
        recall
    """

def selectivity(X, Y, TH=None):
    r"""Compute selectivity or specificity

    .. math::
       {\rm TNR} = {S} = \frac{\rm TN}{{\rm TN} + {\rm FP}}
       :label: equ-selectivity

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    S: float
        selectivity
    """

def fmeasure(X, Y, TH=None, beta=1.0):
    r"""Compute F-measure

    .. math::
       F_{\beta} = \frac{(1+\beta^2)PR}{\beta^2P + R}
       :label: equ-F-measure

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0
    beta : float
        X > TH --> 1, X <= TH --> 0
    Returns
    -------
    F: float
        F-measure
    """

def false_alarm_rate(X, Y, TH=None):
    r"""Compute false alarm rate or False Discovery Rate

    .. math::
       {\rm FDR} = \frac{\rm FP}{{\rm TP} + {\rm FP}} = 1 - P
       :label: equ-FalseDiscoveryRate

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    FDR: float
        False Discovery Rate
    """

def miss_alarm_rate(X, Y, TH=None):
    r"""Compute miss alarm rate or False Negative Rate

    .. math::
       {\rm FNR} = \frac{\rm FN}{{\rm FN} + {\rm TP}} = 1 - R
       :label: equ-FalseNegativeRate

    Parameters
    ----------
    X : Tensor
        retrieval results, retrieved-->1, not retrieved-->0
    Y : Tensor
        referenced, positive-->1, negative-->0
    TH : float
        X > TH --> 1, X <= TH --> 0

    Returns
    -------
    FNR: float
        False Negative Rate
    """


