def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    ...

class GaborConv2d(nn.Module):
    ...

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0):
        ...

    def forward(self, x):
