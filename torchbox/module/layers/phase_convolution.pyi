class PhaseConv1d(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, x):
class PhaseConv2d(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, x):
class ComplexPhaseConv1d(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, x):
class ComplexPhaseConv2d(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    def forward(self, x):
class PhaseConvTranspose1d(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
    def forward(self, x):
class PhaseConvTranspose2d(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
    def forward(self, x):
class ComplexPhaseConvTranspose1d(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
    def forward(self, x):
class ComplexPhaseConvTranspose2d(th.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
    def forward(self, x):