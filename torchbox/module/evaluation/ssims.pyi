class SSIM(torch.nn.Module):
    ...

    def __init__(self, data_range=255, size_average=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, K=(0.01, 0.03), nonnegative_ssim=False):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

    def forward(self, X, Y):
        ...

class MSSSIM(torch.nn.Module):
    ...

    def __init__(self, data_range=255, size_average=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, weights=None, K=(0.01, 0.03)):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

    def forward(self, X, Y):
        ...


