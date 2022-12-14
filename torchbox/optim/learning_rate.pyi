def gammalr(x, k=2, t=2, a=1):
    ...

class LrFinder():
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

        """

        if lrmod in ['log', 'LOG', 'Log']:
            lrs = [math.log10(x) for x in self.lrs]
            lrmod = 'Log10'
            lrunitstr = '/Log10'
        if lrmod in ['linear', 'LINEAR', 'Linear']:
            lrs = self.lrs
            lrmod = ''
            lrunitstr = ''

        if loss in ['smoothed', 'SMOOTHED', 'Smoothed']:
            loss = 'Smoothed'
            losses = self.smt_losses
        if loss in ['average', 'AVERAGE', 'Average']:
            loss = 'Average'
            losses = self.avg_losses
        if loss in ['original', 'ORIGINAL', 'Original']:
            loss = 'Original'
            losses = self.losses

        plt.figure()
        plt.plot(lrs, losses)
        plt.xlabel('Learning rate' + lrunitstr)
        plt.ylabel('Loss')
        plt.grid()

        losslr_str = loss + 'Loss_' + lrmod + 'LR.png'

        if self.plotdir is None:
            plt.show()
        else:
            plt.savefig(self.plotdir + '/' + losslr_str)
            plt.close()

    def find(self, dataloader, model, optimizer, criterion, nin=1, nout=1, nbgc=1, lr_init=1e-8, lr_final=1e2, beta=0.98, gamma=4.):
        r"""Find learning rate


