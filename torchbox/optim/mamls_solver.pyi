class MAML: def __init__(self, net, alpha=0.01):
                ...

    def __init__(self, net, alpha=0.01):
        ...

    def copy_weights(self):
        ...

    def zero_grad(self,):
        ...

    def forward(self, x, adapted_weight=None, **kwards):
        ...

    def update_base(self, grads):
        ...

class MetaSGD: def __init__(self, net):
                   ...

    def __init__(self, net):
        ...

    def copy_weights(self):
        ...

    def zero_grad(self,):
        ...

    def forward(self, x, adapted_weight=None, **kwards):
        ...

    def update_base(self, grads):
        ...

def mamls_train_epoch(mmodel, mdl, criterions, criterionws=None, optimizer=None, scheduler=None, nsteps_base=1, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """train one epoch using MAML, MetaSGD

    Parameters
    ----------
    mmodel : Module
        the network model
    mdl : MetaDataLoader
        the meta dataloader for training :math:`\{(x_s, y_s, x_q, y_q)\}`
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    optimizer : Optimizer or None
        optimizer for meta learner, default is :obj:`None`, 
        which means ``th.optim.Adam(model.parameters(), lr=0.001)``
    scheduler : LrScheduler or None, optional
        scheduler for meta learner, default is :obj:`None`, 
        which means using fixed learning rate
    nsteps_base : int, optional
        the number of fast adapt steps in inner loop, by default 1
    epoch : int or None, optional
        current epoch index, by default None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for training, by default ``'cuda:0'``
    kwargs :
        other forward args
    """

def mamls_valid_epoch(mmodel, mdl, criterions, criterionws=None, nsteps_base=1, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """valid one epoch using MAML, MetaSGD

    Parameters
    ----------
    mmodel : Module
        the network model
    mdl : MetaDataLoader
        the meta dataloader for valid :math:`\{(x_s, y_s, x_q, y_q)\}`
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    nsteps_base : int, optional
        the number of fast adapt steps in inner loop, by default 1
    epoch : int or None, optional
        current epoch index, by default None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for training, by default ``'cuda:0'``
    kwargs :
        other forward args
    """

def mamls_test_epoch(mmodel, mdl, criterions, criterionws=None, nsteps_base=1, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """Test one epoch using MAML, MetaSGD

    Parameters
    ----------
    mmodel : Module
        the network model
    mdl : MetaDataLoader
        the meta dataloader for valid :math:`\{(x_s, y_s, x_q, y_q)\}`
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    nsteps_base : int, optional
        the number of fast adapt steps in inner loop, by default 1
    epoch : int or None, optional
        current epoch index, by default None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for training, by default ``'cuda:0'``
    kwargs :
        other forward args
    """


