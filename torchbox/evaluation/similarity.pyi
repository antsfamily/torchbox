def jaccard_index(X, Y, TH=None):
    r"""Jaccard similarity coefficient

    .. math::
        \mathrm{J}(\mathrm{A}, \mathrm{B})=\frac{|A \cap B|}{|A \cup B|}

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
    JS : float
        the jaccard similarity coefficient.

    """

def dice_coeff(X, Y, TH=0.5):
    r"""Dice coefficient

    .. math::
        s = \frac{2|Y \cap X|}{|X|+|Y|}

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
    DC : float
        the dice coefficient.
    """


