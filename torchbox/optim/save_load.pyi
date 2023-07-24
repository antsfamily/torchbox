def device_transfer(obj, name, device):
    ...

def save_model(modelfile, model, optimizer=None, scheduler=None, epoch=None, mode='parameter'):
    r"""save model to a file

    Parameters
    ----------
    modelfile : str
        model file path
    model : object
        the model object
    optimizer : object or None, optional
        the torch.optim.Optimizer, by default :obj:`None`
    scheduler : object or None, optional
        th.optim.lr_scheduler, by default :obj:`None`
    epoch : int or None, optional
        epoch number, by default :obj:`None`
    mode : str, optional
        save mode, by default ``'parameter'``

    Returns
    -------
    int
        0 is OK
    """

def load_model(modelfile, model, optimizer=None, scheduler=None, mode='parameter', device='cuda:0'):
    r"""load a model from file

    Parameters
    ----------
    modelfile : str
        the model file path
    model : object
        the model object
    optimizer : object or None, optional
        the torch.optim.Optimizer, by default :obj:`None`
    scheduler : object or None, optional
        th.optim.lr_scheduler, by default :obj:`None`
    mode : str, optional
        the mode of saving model, by default ``'parameter'``
    """


