    """

    def __init__(self):
        r"""

        Initialize Binary SPfunction

        """

    def eval(self, v, lmbd):
        r"""eval SP function

        The binary SPL function can be expressed as

        .. math::
           f(\bm{v}, k) =  = -λ\|{\bm v}\|_1 = -λ\sum_{n=1}^N v_n
           :label: equ-SPL_BinaryFunction

        Parameters
        ----------
        v : tensor
            The easy degree of N samples. (:math:`N×1` tensor)
        lmbd : float
            balance factor
        """

    """

    def __init__(self):
        r"""

        Initialize Linear SPfunction

        """

    def eval(self, v, lmbd):
        r"""eval SP function

        The Linear SPL function can be expressed as

        .. math::
           f(\bm{v}, \lambda)=\lambda\left(\frac{1}{2}\|\bm{v}\|_{2}^{2}-\sum_{n=1}^{N} v_{n}\right)
           :label: equ-SPL_LinearFunction

        Parameters
        ----------
        v : tensor
            The easy degree of N samples. (:math:`N×1` tensor)
        lmbd : float
            balance factor
        """

    """

    def __init__(self):
        r"""

        Initialize Logarithmic SPfunction

        """

    def eval(self, v, lmbd):
        r"""eval SP function

        The Logarithmic SPL function can be expressed as

        .. math::
           f(\bm{v}, \lambda) = \sum_{n=1}^{N}\left(\zeta v_{n}-\frac{\zeta^{v_{n}}}{{\rm log} \zeta}\right)
           :label: equ-SPL_LogarithmicFunction

        where, :math:`\zeta=1-\lambda, 0<\lambda<1`

        Parameters
        ----------
        v : tensor
            The easy degree of N samples. (:math:`N×1` tensor)
        lmbd : float
            balance factor
        """

    """

    def __init__(self):
        r"""

        Initialize Mixture SPfunction

        """

    def eval(self, v, lmbd1, lmbd2):
        r"""eval SP function

        The Mixture SPL function can be expressed as

        .. math::
           f\left(\bm{v}, λ \right)=-\zeta \sum_{n=1}^{N} \log \left(v_{n}+\zeta / λ \right)
           :label: equ-SPL_MixtureFunction

        where, :math:`ζ= \frac{1}{k^{\prime} - k} = \frac{\lambda^{\prime}\lambda}{\lambda-\lambda^{\prime}}`


        Parameters
        ----------
        v : tensor
            The easy degree of N samples. (:math:`N×1` tensor)
        """

