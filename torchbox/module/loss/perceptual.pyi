class RandomProjectionLoss(th.nn.Module):
    r"""RandomProjection loss



    """

    def __init__(self, mode='real', baseloss='MSE', channels=[3, 32], kernel_sizes=[(3, 3)], activations=['ReLU'], reduction='mean'):
        ...

    def forward(self, P, G):
        ...

    def weight_init(self):

