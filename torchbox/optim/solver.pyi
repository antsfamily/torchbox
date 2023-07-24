def train_epoch(model, dl, losses, optimizer, scheduler, epoch, logf='terminal', device='cuda:0', **kwargs):
    r"""train one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the training dataloader
    losses : list
        a list torch.nn.Loss instances
    optimizer : function handle
        an instance of torch.optim.Optimizer
    scheduler : function handle
        an instance of torch.optim.LrScheduler
    epoch : int
        epoch index
    logf : str, optional
        IO for print log, file path or 'terminal', by default 'terminal'
    device : str, optional
        device for training, by default 'cuda:0'
    kwargs :
        other forward args
    """

def valid_epoch(model, dl, losses, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    r"""valid one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the validation dataloader
    losses : list
        a list torch.nn.Loss instances
    epoch : int
        epoch index,  default is None
    logf : str, optional
        IO for print log, file path or 'terminal', by default 'terminal'
    device : str, optional
        device for validation, by default 'cuda:0'
    kwargs :
        other forward args
    """

def test_epoch(model, dl, losses, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """Test one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the testing dataloader
    losses : list
        a list torch.nn.Loss instances
    epoch : int or None
        epoch index,  default is None
    logf : str, optional
        IO for print log, file path or 'terminal', by default 'terminal'
    device : str, optional
        device for testing, by default 'cuda:0'
    kwargs :
        other forward args
    """


