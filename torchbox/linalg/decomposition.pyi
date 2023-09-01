def svd_rank(A, svdr='auto'):
    r"""compute rank of the truncated Singular Value Decomposition

    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.

    Parameters
    ----------
    A : Tensor
        The input matrix
    svdr : str or int, optional
        the rank for the truncation, ``'auto'`` for automatic computation, by default ``'auto'``
    """

def eig(A, cdim=None, dim=(-2, -1), keepdim=False):
    r"""Computes the eigenvalues and eigenvectors of a square matrix.

    Parameters
    ----------
    A : Tensor
        any size tensor, both complex and real representation are supported.
        For real representation, the real and imaginary dimension is specified by :attr:`cdim` or :attr:`caxis`.
    cdim : int or None, optional
        if :attr:`A` and :attr:`B` are complex tensors but represented in real format, :attr:`cdim` or :attr:`caxis`
        should be specified (Default is :obj:`None`).
    dim : tulpe or list
        dimensions for multiplication (default is (-2, -1))
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    """

def eigvals(A, cdim=None, dim=(-2, -1), keepdim=False):
    """Computes the eigenvalues of a square matrix.

    Parameters
    ----------
    A : Tensor
        any size tensor, both complex and real representation are supported.
        For real representation, the real and imaginary dimension is specified by :attr:`cdim` or :attr:`caxis`.
    cdim : int or None, optional
        if :attr:`A` and :attr:`B` are complex tensors but represented in real format, :attr:`cdim` or :attr:`caxis`
        should be specified (Default is :obj:`None`).
    dim : tulpe or list
        dimensions for multiplication (default is (-2, -1))
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    """

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """

    def forward(ctx, input):
        ...

    def backward(ctx, grad_output):
        ...


