class TotalVariation(th.nn.Module):
    r"""Total Variarion

           # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)

    """

    def __init__(self, axis=0, reduction='mean'):
        ...

    def forward(self, X):
        ...


