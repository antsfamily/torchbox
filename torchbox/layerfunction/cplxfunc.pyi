def csign(x, cdim=None):
    r"""The signum function like Matlab's sign

    .. math::
        {\rm csign}(x+jy) = \frac{x+jy}{\sqrt{x^2+y^2}}

    Parameters
    ----------
    x : Tensor, int, float or complex
        The input
    cdim : int or None, optional
        Specifies the complex axis..

    Returns
    -------
    tensor
        The output.

    Raises
    ------
    TypeError
        :attr:`cdim` should be integer!
    """

def csoftshrink(x, alpha=0.5, cdim=None, inplace=False):
    r"""Complex soft shrink function

    Parameters
    ----------
    x : Tensor
        The input.
    alpha : float, optional
        The threshhold.
    cdim : int or None, optional
        Specifies the complex axis.

    Returns
    -------
    tensor
        The output.

    Raises
    ------
    TypeError
        :attr:`cdim` should be integer!
    """

def softshrink(x, alpha=0.5, inplace=False):
    r"""Real soft shrink function

    Parameters
    ----------
    x : Tensor
        The input.
    alpha : float, optional
        The threshhold.

    Returns
    -------
    tensor
        The output.
    """


