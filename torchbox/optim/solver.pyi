def train_epoch(model, dl, criterions, criterionws=None, optimizer=None, scheduler=None, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    r"""train one epoch

    Parameters
    ----------
    model : Module
        an instance of torch.nn.Module
    dl : DataLoader
        the dataloader for training
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    optimizer : Optimizer or None
        an instance of torch.optim.Optimizer, default is :obj:`None`, 
        which means ``th.optim.Adam(model.parameters(), lr=0.001)``
    scheduler : LrScheduler or None
        an instance of torch.optim.LrScheduler, default is :obj:`None`, 
        which means using fixed learning rate
    epoch : int
        epoch index
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for training, by default ``'cuda:0'``
    kwargs :
        other forward args

    see also :func:`~torchbox.optim.solver.valid_epoch`, :func:`~torchbox.optim.solver.test_epoch`, :func:`~torchbox.optim.save_load.save_model`, :func:`~torchbox.optim.save_load.load_model`.
        
    """

def valid_epoch(model, dl, criterions, criterionws=None, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    r"""valid one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the validation dataloader
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    epoch : int
        epoch index,  default is None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for validation, by default ``'cuda:0'``
    kwargs :
        other forward args

    see also :func:`~torchbox.optim.solver.train_epoch`, :func:`~torchbox.optim.solver.test_epoch`, :func:`~torchbox.optim.save_load.save_model`, :func:`~torchbox.optim.save_load.load_model`.

    """

def test_epoch(model, dl, criterions, criterionws=None, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """Test one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the testing dataloader
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    epoch : int or None
        epoch index,  default is None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for testing, by default ``'cuda:0'``
    kwargs :
        other forward args

    see also :func:`~torchbox.optim.solver.train_epoch`, :func:`~torchbox.optim.solver.valid_epoch`, :func:`~torchbox.optim.save_load.save_model`, :func:`~torchbox.optim.save_load.load_model`.

    """


