class Binary(object):
    r"""binary function

    The binary SPL function can be expressed as

    .. math::
       f(\bm{v}, k) =  = -lambd\|{\bm v}\|_1 = -lambd\sum_{n=1}^N v_n
       :label: equ-SPL_BinaryFunction

    The optimal solution is

    .. math::
       v_{n}^* = \left\{\begin{array}{ll}{1,} & {l_{n}<\lambda} \\ {0,} & {l_{n}>=\lambda}\end{array}\right.
       :label: equ-SPL_BinaryUpdate

    """

        """
        super(Binary, self).__init__()
        self.spfunction = 'Binary'
        self.lmbd = -1
        self.mu = mu
        self.rankr = rankr
        self.maxrankr = maxrankr
        self.v = -1

    def step(self, loss):
        r"""one step of optimization

        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

class Linear(object):
    r"""Linear function

    The Linear SPL function can be expressed as

    .. math::
       f(\bm{v}, \lambda)=\lambda\left(\frac{1}{2}\|\bm{v}\|_{2}^{2}-\sum_{n=1}^{N} v_{n}\right)
       :label: equ-SPL_LinearFunction

    The optimal solution is

    .. math::
       v_{n}^* = {\rm max}\{1-l_n/\lambda, 0\}
       :label: equ-SPL_LinearUpdate

    """

        """
        super(Linear, self).__init__()
        self.spfunction = 'Linear'
        self.lmbd = -1
        self.mu = mu
        self.rankr = rankr
        self.maxrankr = maxrankr
        self.v = -1

    def step(self, loss):
        r"""one step of optimization

        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

class Logarithmic(object):
    r"""Logarithmic function

    The Logarithmic SPL function can be expressed as

    .. math::
       f(\bm{v}, \lambda) = \sum_{n=1}^{N}\left(\zeta v_{n}-\frac{\zeta^{v_{n}}}{{\rm log} \zeta}\right)
       :label: equ-SPL_LogarithmicFunction

    where, :math:`\zeta=1-\lambda, 0<\lambda<1`

    The optimal solution is

    .. math::
       v_{n}^{*}=\left\{\begin{array}{ll}{0,} & {l_{n}>=\lambda} \\ {\log \left(l_{n}+\zeta\right) / \log \xi,} & {l_{n}<\lambda}\end{array}\right.
       :label: equ-SPL_LogarithmicUpdate

    """

        """
        super(Logarithmic, self).__init__()
        self.spfunction = 'Logarithmic'
        self.lmbd = -1
        self.mu = mu
        self.rankr = rankr
        self.maxrankr = maxrankr
        self.v = -1

    def step(self, loss):
        r"""one step of optimization

        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """

class Mixture(object):
    r"""Mixture function

    The Mixture SPL function can be expressed as

    .. math::
       f\left(\bm{v}, lambd \right)=-\zeta \sum_{n=1}^{N} \log \left(v_{n}+\zeta / lambd \right)
       :label: equ-SPL_MixtureFunction

    where, :math:`ζ= \frac{1}{k^{\prime} - k} = \frac{\lambda^{\prime}\lambda}{\lambda-\lambda^{\prime}}`

    The optimal solution is

    .. math::
       v_{n}^{*}=\left\{\begin{array}{ll}{1,} & {l_{n} \leq \lambda^{\prime}} \\ {0,} & {l_{n} \geq \lambda} \\ {\zeta / l_{n}-\zeta / \lambda,} & {\text { otherwise }}\end{array}\right.
       :label: equ-SPL_MixtureUpdate

    """

        """
        super(Mixture, self).__init__()
        self.spfunction = 'Mixture'
        self.lmbd = -1
        self.lmbd = -1
        self.mu = mu
        self.rankr = rankr
        self.maxrankr = maxrankr
        self.v = -1

    def step(self, loss):
        r"""one step of optimization

        r"""update rank ratio

        .. math::
           r = {\rm min}\{r*\mu, r_{max}\}
        """


