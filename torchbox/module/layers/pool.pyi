class MeanSquarePool2d(th.nn.Module):
    ...

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        ...

    def forward(self, x):
        ...

class PnormPool2d(th.nn.Module):
    ...

    def __init__(self, kernel_size, p=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        ...

    def forward(self, x):

