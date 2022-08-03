def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """

def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    
    Returns:
        torch.Tensor: blurred tensors
    """

def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    r""" Calculate ssim index for X and Y
    
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    
    Returns:
        torch.Tensor: ssim results.
    """

def ssim( X,
    r""" interface of ssim
    
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size (int, optional): the size of gauss kernel
        win_sigma (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    
    Returns:
        torch.Tensor: ssim results
    """

def msssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)):
    r""" interface of ms-ssim

    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size (int, optional): the size of gauss kernel
        win_sigma (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    
    Returns:
        torch.Tensor: msssim results
    """


