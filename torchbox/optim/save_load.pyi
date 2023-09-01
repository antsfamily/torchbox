def device_transfer(obj, name, device):
    ...

def save_model(modelfile, model, optimizer=None, scheduler=None, epoch=None, mode='parameter'):
    r"""save model to a file

    Parameters
    ----------
    modelfile : str
        model file path
    model : object
        the model object or parameter dict
    optimizer : object or None, optional
        the torch.optim.Optimizer, by default :obj:`None`
    scheduler : object or None, optional
        th.optim.lr_scheduler, by default :obj:`None`
    epoch : int or None, optional
        epoch number, by default :obj:`None`
    mode : str, optional
        saving mode, ``'model'`` means saving model structure and parameters,
        ``'parameter'`` means only saving parameters (default)

    Returns
    -------
    int
        0 is OK
    """

def load_model(modelfile, model=None, optimizer=None, scheduler=None, mode='parameter', device='cpu'):
    r"""load a model from file

    Parameters
    ----------
    modelfile : str
        the model file path
    model : object or None
        the model object or :obj:`None` (default)
    optimizer : object or None, optional
        the torch.optim.Optimizer, by default :obj:`None`
    scheduler : object or None, optional
        th.optim.lr_scheduler, by default :obj:`None`
    mode : str, optional
        the saving mode of model in file, ``'model'`` means saving model structure and parameters,
        ``'parameter'`` means only saving parameters (default)
    device : str, optional
        load model to the specified device
    """

def get_parameters(model, optimizer=None, scheduler=None, epoch=None):
    r"""save model to a file

    Parameters
    ----------
    model : object
        the model object
    optimizer : object or None, optional
        the torch.optim.Optimizer, by default :obj:`None`
    scheduler : object or None, optional
        th.optim.lr_scheduler, by default :obj:`None`
    epoch : int or None, optional
        epoch number, by default :obj:`None`

    Returns
    -------
    dict
        keys: 'epoch', 'network' (model.state_dict), 'optimizer' (optimizer.state_dict), 'scheduler' (scheduler.state_dict)
    """


