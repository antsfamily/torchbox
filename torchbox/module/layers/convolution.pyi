class FFTConv1(th.nn.Module):
    def __init__(self, nh, h=None, axis=0, nfft=None, shape='same', train=True):
    def forward(self, x):
class Conv1(th.nn.Module):
    def __init__(self, axis, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, X):
class MaxPool1(th.nn.Module):
    def __init__(self, axis, kernel_size, stride=None, padding=0,
    def forward(self, X):
class Conv2(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, X):
class MaxPool2(th.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
    def forward(self, X):