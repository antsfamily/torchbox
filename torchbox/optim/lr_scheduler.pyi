class GaussianLR(th.optim.lr_scheduler._LRScheduler):
    r"""GaussianLR

    Set the learning rate of each parameter group using a double gaussian kernel schedule

    .. image:: ./_static/GaussianLREquation.png
       :scale: 50 %
       :align: center

    where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators.

    The maximum learning rate are the base learning rate setted in Optimizer.


    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    t_eta_max : int
        Iterations when the learning rate reach to the maximum value :math:`\eta_{\max}`.
    sigma1 : int
        Controls the shape of warming up phase.
    sigma2 : int
        Controls the shape of annealing phase.
    eta_start : float
        Starting learning rate. Default: 0.
    eta_stop : float
        Stopping learning rate. Default: 0.
    last_epoch : int
        The index of last epoch. Default: -1.

    Examples
    ---------

    .. image:: ./_static/DoubleGaussianKernelLR.png
       :scale: 50 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torch as th
        import torchbox as tb
        import matplotlib; matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        lr = 1e-1
        lr = 1e-2
        # lr = 1e2

        num_epochs = 1000
        num_epochs = 500
        batch_size = 8
        num_batch = 750

        params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
                th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
                }

        optimizer = th.optim.Adam(params, lr=lr)
        # optimizer = th.optim.SGD(params, lr=lr, momentum=0.9)
        scheduler = tb.optim.lr_scheduler.GaussianLR(optimizer, t_eta_max=50, sigma1=15, sigma2=100, eta_start=1e-4, eta_stop=1e-3, last_epoch=-1)

        print(optimizer)

        lrs = []
        for n in range(num_epochs):
            for b in range(num_batch):

                optimizer.step()

                # lrs.append(optimizer.param_groups[0]['lr'])

            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        plt.figure()
        plt.plot(lrs)
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.grid()
        plt.show()

    """

    def __init__(self, optimizer, t_eta_max, sigma1, sigma2, eta_start=1e-6, eta_stop=1e-5, last_epoch=-1):
        ...

    def get_lr(self):
        ...

    def _get_closed_form_lr(self):
        ...

class MountainLR(th.optim.lr_scheduler._LRScheduler):
    r"""MountainLR
    
    Set the learning rate of each parameter group using a double gaussian kernel

    .. math::
        (|x-P| / N) .* (-2 + cos(2 * (x-P) / T))

    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators.

    The maximum learning rate are the base learning rate setted in Optimizer.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    t_eta_max : int
        Iterations when the learning rate reach to the maximum value :math:`\eta_{\max}`.
    sigma1 : int
        Controls the shape of warming up phase.
    sigma2 : int
        Controls the shape of annealing phase.
    eta_start : float
        Starting learning rate. Default: 0.
    eta_stop : float
        Stopping learning rate. Default: 0.
    last_epoch : int
        The index of last epoch. Default: -1.

    Examples
    ---------

    .. image:: ./_static/MountainLR.png
       :scale: 50 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torch as th
        import torchbox as tb
        import matplotlib; matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        lr = 1e-1
        lr = 1e-2
        # lr = 1e2

        num_epochs = 1000
        num_epochs = 500
        batch_size = 8
        num_batch = 750

        params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
                th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
                }

        optimizer = th.optim.Adam(params, lr=lr)
        scheduler = tb.optim.lr_scheduler.MountainLR(optimizer, total_epoch=num_epochs, peak_epoch=300, period_epoch=50, last_epoch=-1)

        print(optimizer)

        lrs = []
        for n in range(num_epochs):
            for b in range(num_batch):

                optimizer.step()

                # lrs.append(optimizer.param_groups[0]['lr'])

            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        plt.figure()
        plt.plot(lrs)
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.grid()
        plt.show()

    """

    def __init__(self, optimizer, total_epoch, peak_epoch, period_epoch, last_epoch=-1):
        ...

    def get_lr(self):
        ...

    def _get_closed_form_lr(self):
        ...


