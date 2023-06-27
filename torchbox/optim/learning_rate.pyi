def gammalr(x, k=2, t=2, a=1):
    ...

class LrFinder():
    ...

    def __init__(self, device='cpu', plotdir=None, logf=None):
        r"""init

        Initialize LrFinder.

        Parameters
        ----------
        device : str, optional
            device string: ``'cpu'``(default), ``'cuda:0'``, ``cuda:1`` ...
        plotdir : str, optional
            If it is not None, plot the loss-lr curve and save the figure,
            otherwise plot and show but not save. (the default is None).
        logf : str or None optional
            print log to terminal or file.
        """

    def plot(self, lrmod='log', loss='smoothed'):
        r"""plot the loss-lr curve

        Plot the loss-learning rate curve.

        Parameters
        ----------
        lrmod : str, optional
            ``'log'`` --> use log scale, i.e. log10(lr) instead lr. (default)
            ``'linear'`` --> use original lr.
        loss : str, optional
            Specify which type of loss will be ploted. (the default is 'smoothed')
        """

    def find(self, dataloader, model, optimizer, criterion, nin=1, nout=1, nbgc=1, lr_init=1e-8, lr_final=1e2, beta=0.98, gamma=4.):
        r"""Find learning rate

        Find learning rate, see `How Do You Find A Good Learning Rate <https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html>`_ .

        During traing, two types losses are computed

        The average loss is:

        .. math::
           \rm{avg\_loss}_i=\beta * \rm{avg\_loss}_{i-1}+(1-\beta) * \rm{loss}_i

        The smoothed loss is:

        .. math::
            \rm{smt\_loss }_{i}=\frac{\rm{avg\_loss}_{i}}{1-\beta^{i+1}}

        If :math:`i > 1` and :math:`\rm{smt\_loss} > \gamma * \rm{best\_loss}`, stop.

        If :math:`\rm{smt\_loss} < \rm{best\_loss}` or :math:`i = 1`, let :math:`\rm{best\_loss} = \rm{smt\_loss}`.


        Parameters
        ----------
        dataloader : DataLoader
            The dataloader that contains a dataset for training.
        model : Module
            Your network module.
        optimizer : Optimizer
            The optimizer such as SGD, Adam...
        criterion : Loss
            The criterion/loss used for training model.
        nin : int, optional
            The number of inputs of the model,
            the first :attr:`nin` elements are inputs,
            the rest are targets(can be None) used for computing loss. (the default is 1)
        nou : int, optional
            The number of outputs of the model used for computing loss,
            it works only when the model has multiple outputs, i.e.
            the outputs is a tuple or list which has several tensor elements (>=1).
            the first :attr:`nout` elements are used for computing loss,
            the rest are ignored. (the default is 1)
        nbgc : int, optional
            The number of batches for grad cumulation (the default is 1, which means no cumulation)
        lr_init : int, optional
            The initial learning rate (the default is 1e-8)
        lr_final : int, optional
            The final learning rate (the default is 1e-8)
        beta : float, optional
            weight for weighted sum of loss (the default is 0.98)
        gamma : float, optional
            The exploding factor :math:`\gamma`. (the default is 4.)


        Returns
        -------
        lrs : list
            Learning rates during training.
        smt_losses : list
            Smoothed losses during training.
        avg_losses : list
            Average losses during training.
        losses : list
            Original losses during training.

        Examples
        --------

        ::

            device = 'cuda:1'
            # device = 'cpu'

            num_epochs = 30
            X = th.randn(100, 2, 3, 4)
            Y = th.randn(100, 1, 3, 4)

            trainds = TensorDataset(X, Y)
            # trainds = TensorDataset(X)

            model = th.nn.Conv2d(2, 1, 1)
            model.to(device)

            trainld = DataLoader(trainds, batch_size=10, shuffle=False)

            criterion = th.nn.MSELoss(reduction='mean')

            optimizer = th.optim.SGD(model.parameters(), lr=1e-1)

            lrfinder = LrFinder(device)
            # lrfinder = LrFinder(device, plotdir='./')

            lrfinder.find(trainld, model, optimizer, criterion, nin=1,
                          nbgc=1, lr_init=1e-8, lr_final=10., beta=0.98)

            lrfinder.plot(lrmod='Linear')
            lrfinder.plot(lrmod='Log')
        """


