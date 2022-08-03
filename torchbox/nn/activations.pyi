def linear(x):
    r"""linear activation

    .. math::
       y = x

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def sigmoid(x):
    r"""sigmoid function

    .. math::
        y = \frac{e^x}{e^x + 1}

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def tanh(x):
    r"""tanh function

    .. math::
        y = {\rm tanh}(x) = {{e^{2x} - 1} \over {e^{2x} + 1}}.

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def softplus(x):
    r"""softplus function

    .. math::
       {\rm log}(e^x + 1)

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def softsign(x):
    r"""softsign function

    .. math::
       \frac{x} {({\rm abs}(x) + 1)}

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def elu(x):
    r"""Computes exponential linear element-wise.

    .. math::
        y = \left\{ {\begin{tensor}{*{20}{c}}{x,\;\;\;\;\;\;\;\;\;x \ge 0}\\{{e^x} - 1,\;\;\;x < 0}\end{tensor}} \right..

    See  `Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) <http://arxiv.org/abs/1511.07289>`_  

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def relu(x):
    r"""Computes rectified linear

    .. math::
       {\rm max}(x, 0)

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def relu6(x):
    r"""Computes Rectified Linear 6

    .. math::
       {\rm min}({\rm max}(x, 0), 6)

    `Convolutional Deep Belief Networks on CIFAR-10. A. Krizhevsky <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_  

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def selu(x):
    r"""Computes scaled exponential linear

    .. math::
        y = \lambda \left\{ {\begin{tensor}{*{20}{c}}{x, x \ge 0}\\{\alpha ({e^x} - 1), x < 0}\end{tensor}} \right.
    
    where, :math:`\alpha = 1.6732632423543772848170429916717` , :math:`\lambda = 1.0507009873554804934193349852946`, 
    See `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def crelu(x):
    r"""Computes Concatenated ReLU.

    Concatenates a ReLU which selects only the positive part of the activation
    with a ReLU which selects only the *negative* part of the activation.
    Note that as a result this non-linearity doubles the depth of the activations.
    Source: `Understanding and Improving Convolutional Neural Networks via
    Concatenated Rectified Linear Units. W. Shang, et
    al. <https://arxiv.org/abs/1603.05201>`_

    Parameters
    ----------
    x : lists or tensor
        inputs

    Returns
    -------
    tensor
        outputs
    """

def leaky_relu(x, alpha=0.2):
    r"""Compute the Leaky ReLU activation function. 

    .. math::
        y = \left\{ {\begin{tensor}{ccc}{x, x \ge 0}\\{\alpha x, x < 0}\end{tensor}} \right.

    `Rectifier Nonlinearities Improve Neural Network Acoustic Models <http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf>`_  

    Parameters
    ----------
    x : lists or tensor
        inputs
    alpha : float
        :math:`\alpha`

    Returns
    -------
    tensor
        outputs
    """

def swish(x, beta=1.0):
    r"""Swish function

    .. math::
       y = x\cdot {\rm sigmoid}(\beta x) = {e^{(\beta x)} \over {e^{(\beta x)} + 1}} \cdot x

    See `"Searching for Activation Functions" (Ramachandran et al. 2017) <https://arxiv.org/abs/1710.05941>`_  

    Parameters
    ----------
    x : lists or tensor
        inputs
    beta : float
        :math:`\beta`

    Returns
    -------
    tensor
        outputs
    """


