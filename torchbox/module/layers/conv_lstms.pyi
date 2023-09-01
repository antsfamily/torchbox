class ConvLSTMCell(th.nn.Module):
    r"""Cell class for the ConvLSTM layer

    Convolutional LSTM Cell.

    Parameters
    ----------
    rank : int
        1 for 1D convolution, 2 for 2D convolution, 3 for 3D convolution
    in_channels : int
        the number of input channels
    out_channels : int
        the number of output channels
    kernel_size : int, list or tuple
        the window size of convolution
    stride : int, optional
        the stride of convolution, by default 1
    padding : int, str, optional
        ``'valid'``, ``'same'``, by default ``'same'``
    dilation : int, optional
        Spacing between kernel elements. Default: 1
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias : bool, optional
        If ``'True'``, adds a learnable bias to the output, by default ``'True'``
    padding_mode : str, optional
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    activation : str or None, optional
        activation of input convolution layer, ``'Tanh()'`` (default), ``'Sigmoid'``, ...
    rnn_activation : str or None, optional
        activation of RNN convolution layer, ``'Hardsigmoid()'`` (default), ``'Sigmoid'``, ...
    dropp : float or None, optional
        dropout rate of input convolution layer, :obj:`None`
    rnn_dropp : float or None, optional
        dropout rate of RNN layer, :obj:`None`
    device : str or None, optional
        device for computation, by default obj:`None`
    dtype : str or None, optional
        data type, by default None

    Returns
    --------
    h : Tensor
        hidden states
    c : Tensor
        code states
    
    """

    def __init__(self, rank, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', activation='Tanh()', rnn_activation='Hardsigmoid()', dropp=None, rnn_dropp=None, device=None, dtype=None):
        ...

    def get_hc_shape(self, xshape):
        ...

    def forward(self, x, states):
        ...

class ConvLSTM(th.nn.Module):
    r"""class for the ConvLSTM layer

    Convolutional LSTM.
    
    input shape: (B, T, C, L) or (B, T, C, H, W) or (B, T, C, H, W, K)

    Parameters
    ----------
    rank : int
        1 for 1D convolution, 2 for 2D convolution, 3 for 3D convolution
    in_channels : int, list or tuple
        the number of input channels of each cell
    out_channels : int, list or tuple
        the number of output channels of each cell
    kernel_size : int, list or tuple
        the window size of convolution of each cell
    stride : int, list or tuple, optional
        the stride of convolution of each cell, by default 1
    padding : int, str, list or tuple, optional
        the padding size of convolution of each cell, ``'valid'``, ``'same'``, by default ``'same'``
    dilation : int, list or tuple, optional
        the spacing between kernel elements of each cell, by default 1
    groups : int, list or tuple, optional
        the number of blocked connections from input channels to output channels of each cell, by default 1
    bias : bool, list or tuple, optional
        If :obj:`True`, adds a learnable bias to the output of convolution of each cell, by default :obj:`True`
    padding_mode : str, list or tuple, optional
        'zeros', 'reflect', 'replicate' or 'circular', by default 'zeros'
    activation : str or None, optional
        activation of input convolution layers, ``'Tanh()'`` (default), ``'Sigmoid'``, ...
    rnn_activation : str or None, optional
        activation of RNN convolution layers, ``'Hardsigmoid()'`` (default), ``'Sigmoid'``, ...
    dropp : float or None, optional
        dropout rate of input convolution layers, :obj:`None`
    rnn_dropp : float or None, optional
        dropout rate of RNN layers, :obj:`None`
    bidirectional : bool, optional
        :obj:`True` for bidirectional convolutional LSTM, by default :obj:`False`
    batch_first : bool, optional
        :obj:`True` for ``(B, T, ...)``, by default :obj:`False`
    device : str or None, optional
        device for computation, by default None
    dtype : str or None, optional
        data type, by default None

    Returns
    ----------
    xs : Tensor
        output sequence
    states : tuple of list
        (hidden states, code states) of each cell

    Examples
    ----------

    Stack two Conv2dLSTM Cells with ConvLSTM and ConvLSTMCell, respectively.
    
    ::

        import torch as th
        import torchbox as tb


        tb.setseed(seed=2023, target='torch')

        T, B, C, H, W = 10, 6, 2, 18, 18
        x = th.randn(T, B, C, H, W)

        # ===way1
        tb.setseed(seed=2023, target='torch')
        lstm = tb.ConvLSTM(rank=2, in_channels=[C, 4], out_channels=[4, 4], kernel_size=[3, 3], stride=[1, 1], padding=['same', 'same'])
        print(lstm.cells[0].in_convc.weight.sum(), lstm.cells[0].rnn_convc.weight.sum())
        print(lstm.cells[1].in_convc.weight.sum(), lstm.cells[1].rnn_convc.weight.sum())

        # ===way2
        tb.setseed(seed=2023, target='torch')
        cell1 = tb.ConvLSTMCell(rank=2, in_channels=C, out_channels=4, kernel_size=3, stride=1, padding='same')
        cell2 = tb.ConvLSTMCell(rank=2, in_channels=4, out_channels=4, kernel_size=3, stride=1, padding='same')

        print(cell1.in_convc.weight.sum(), cell1.rnn_convc.weight.sum())
        print(cell2.in_convc.weight.sum(), cell2.rnn_convc.weight.sum())

        # ===way1
        y, (h, c) = lstm(x, None)
        h = th.stack(h, dim=0)
        c = th.stack(c, dim=0)

        print(y.shape, y.sum(), h.shape, h.sum(), c.shape, c.sum())

        # ===way2
        h1, c1 = None, None
        h2, c2 = None, None
        y = []
        for t in range(x.shape[0]):
            h1, c1 = cell1(x[t, ...], (h1, c1))
            h2, c2 = cell2(h1, (h2, c2))
            y.append(h2)
        y = th.stack(y, dim=0)
        h = th.stack((h1, h2), dim=0)
        c = th.stack((c1, c2), dim=0)
        print(y.shape, y.sum(), h.shape, h.sum(), c.shape, c.sum())


        # output
        tensor(-1.4177, grad_fn=<SumBackward0>) tensor(0.9743, grad_fn=<SumBackward0>)
        tensor(0.1532, grad_fn=<SumBackward0>) tensor(-0.1598, grad_fn=<SumBackward0>)
        tensor(-1.4177, grad_fn=<SumBackward0>) tensor(0.9743, grad_fn=<SumBackward0>)
        tensor(0.1532, grad_fn=<SumBackward0>) tensor(-0.1598, grad_fn=<SumBackward0>)
        torch.Size([10, 6, 4, 18, 18]) tensor(-2144.8628, grad_fn=<SumBackward0>) torch.Size([2, 6, 4, 18, 18]) tensor(-398.1468, grad_fn=<SumBackward0>) torch.Size([2, 6, 4, 18, 18]) tensor(-783.8212, grad_fn=<SumBackward0>)
        torch.Size([10, 6, 4, 18, 18]) tensor(-2144.8628, grad_fn=<SumBackward0>) torch.Size([2, 6, 4, 18, 18]) tensor(-398.1468, grad_fn=<SumBackward0>) torch.Size([2, 6, 4, 18, 18]) tensor(-783.8212, grad_fn=<SumBackward0>)
    
    """

    def __init__(self, rank, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', activation='Tanh()', rnn_activation='Hardsigmoid()', dropp=None, rnn_dropp=None, bidirectional=False, batch_first=False, device=None, dtype=None):
        ...

    def get_hc_shape(self, xshape):
        ...

    def forward(self, x, states=None):
        ...


