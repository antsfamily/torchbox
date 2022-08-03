def rect(x):
    r"""
    Rectangle function:

    .. math::
        rect(x) = {1, if |x|<= 0.5; 0, otherwise}
    """

def chirp(t, T, Kr):
    r"""
    Create a chirp signal:

    .. math::
        S_{tx}(t) = rect(t/T) * exp(1j*pi*Kr*t^2)
    """


