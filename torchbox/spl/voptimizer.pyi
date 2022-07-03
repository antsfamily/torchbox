    """

    def __init__(self, rankr=0.6, maxrankr=1, mu=1.003):
        r"""

        Initialize Binary optimizer

        Parameters
        ----------
        rankr : float, optional
            the initial proportion :math:`r` of the selected samples (with weights vi=1); (the default is 0.6)
        maxrankr : int, optional
            the upper bound of the annealed sample proportion :math:`r_{max}`. (the default is 1)
        mu : int, optional
            the annealing parameter :math`\mu`, the incremental ratio of the proportion of the selected
            samples in each iteration. (the default is 1.003)
        """

    def step(self, loss):
        r"""one step of optimization

        The optimal solution is

        .. math::
           v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
           :label: equ-SPL_BinaryUpdate

        Parameters
        ----------
        loss : tensor
            The loss values of N samples. (:math:`N×1` tensor)
        """

    def update_rankr(self):
        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

    """

    def __init__(self, rankr=0.6, maxrankr=1, mu=1.003):
        r"""

        Initialize Linear optimizer

        Parameters
        ----------
        rankr : float, optional
            the initial proportion :math:`r` of the selected samples (with weights vi=1); (the default is 0.6)
        maxrankr : int, optional
            the upper bound of the annealed sample proportion :math:`r_{max}`. (the default is 1)
        mu : int, optional
            the annealing parameter :math`\mu`, the incremental ratio of the proportion of the selected
            samples in each iteration. (the default is 1.003)
        """

    def step(self, loss):
        r"""one step of optimization

        The optimal solution is

        .. math::
           v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
           :label: equ-SPL_BinaryUpdate

        Parameters
        ----------
        loss : tensor
            The loss values of N samples. (:math:`N×1` tensor)
        """

    def update_rankr(self):
        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

    """

    def __init__(self, rankr=0.6, maxrankr=1, mu=1.003):
        r"""

        Initialize Logarithmic optimizer

        Parameters
        ----------
        rankr : float, optional
            the initial proportion :math:`r` of the selected samples (with weights vi=1); (the default is 0.6)
        maxrankr : int, optional
            the upper bound of the annealed sample proportion :math:`r_{max}`. (the default is 1)
        mu : int, optional
            the annealing parameter :math`\mu`, the incremental ratio of the proportion of the selected
            samples in each iteration. (the default is 1.003)
        """

    def step(self, loss):
        r"""one step of optimization

        The optimal solution is

        .. math::
           v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
           :label: equ-SPL_BinaryUpdate

        Parameters
        ----------
        loss : tensor
            The loss values of N samples. (:math:`N×1` tensor)
        """

    def update_rankr(self):
        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

    """

    def __init__(self, rankr=0.6, maxrankr=1, mu=1.003):
        r"""

        Initialize Mixture optimizer

        Parameters
        ----------
        rankr : float, optional
            the initial proportion :math:`r` of the selected samples (with weights vi=1); (the default is 0.6)
        maxrankr : int, optional
            the upper bound of the annealed sample proportion :math:`r_{max}`. (the default is 1)
        mu : int, optional
            the annealing parameter :math`\mu`, the incremental ratio of the proportion of the selected
            samples in each iteration. (the default is 1.003)
        """

    def step(self, loss):
        r"""one step of optimization

        The optimal solution is

        .. math::
           v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
           :label: equ-SPL_BinaryUpdate

        Parameters
        ----------
        loss : tensor
            The loss values of N samples. (:math:`N×1` tensor)
        """

    def update_rankr(self):
        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

