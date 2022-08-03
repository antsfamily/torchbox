def orth(x):
    r"""Orthogonalization

    A function like MATLAB's ``orth``. After orthogonalizing,
    each column is a orthogonal basis.

    Parameters
    ----------
    x : Tensor
        The matrix to be orthogonalized.

    Examples
    --------

    code:
    ::

        x = th.tensor([[1, 2.], [3, 4], [5, 6]])
        y = orth(x)
        print(x)
        print(y)
        print((y[0, :] * y[1, :] * y[2, :]).sum())
        print((y[:, 0] * y[:, 1]).sum())

    result:
    ::

        tensor([[1., 2.],
                [3., 4.],
                [5., 6.]])
        tensor([[-0.2298,  0.8835],
                [-0.5247,  0.2408],
                [-0.8196, -0.4019]])
        tensor(-0.1844)
        tensor(-1.7881e-07)

    """


