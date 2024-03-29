import torch as th
import torch.nn.functional as F
from collections import OrderedDict


N_FILTERS = 64  # number of filters used in conv_block
K_SIZE = 3  # size of kernel
MP_SIZE = 2  # size of max pooling
EPS = 1e-8  # epsilon for numerical stability

class Net(th.nn.Module):
    """
    The base CNN model for MAML (Meta-SGD) for few-shot learning.
    The architecture is same as of the embedding in MatchingNet.
    """

    def __init__(self, in_channels, num_classes, dataset='Omniglot'):
        """
        self.net returns:
            [N, 64, 1, 1] for Omniglot (28x28)
            [N, 64, 5, 5] for miniImageNet (84x84)
        self.fc returns:
            [N, num_classes]
        
        Args:
            in_channels: number of input channels feeding into first conv_block
            num_classes: number of classes for the task
            dataset: for the measure of input units for self.fc, caused by 
                     difference of input size of 'Omniglot' and 'ImageNet'
        """
        super(Net, self).__init__()
        self.features = th.nn.Sequential(
            conv_block(0, in_channels, padding=1, pooling=True),
            conv_block(1, N_FILTERS, padding=1, pooling=True),
            conv_block(2, N_FILTERS, padding=1, pooling=True),
            conv_block(3, N_FILTERS, padding=1, pooling=True))

    def forward(self, X, params=None):

        if params == None:
            out = self.features(X)
        else:
            """
            The architecure of functionals is the same as `self`.
            """
            out = F.conv2d(
                X,
                params['meta_learner.features.0.conv0.weight'],
                params['meta_learner.features.0.conv0.bias'],
                padding=1)
            # NOTE we do not need to care about running_mean anv var since
            # momentum=1.
            out = F.batch_norm(
                out,
                params['meta_learner.features.0.bn0.running_mean'],
                params['meta_learner.features.0.bn0.running_var'],
                params['meta_learner.features.0.bn0.weight'],
                params['meta_learner.features.0.bn0.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.1.conv1.weight'],
                params['meta_learner.features.1.conv1.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.1.bn1.running_mean'],
                params['meta_learner.features.1.bn1.running_var'],
                params['meta_learner.features.1.bn1.weight'],
                params['meta_learner.features.1.bn1.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.2.conv2.weight'],
                params['meta_learner.features.2.conv2.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.2.bn2.running_mean'],
                params['meta_learner.features.2.bn2.running_var'],
                params['meta_learner.features.2.bn2.weight'],
                params['meta_learner.features.2.bn2.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = F.conv2d(
                out,
                params['meta_learner.features.3.conv3.weight'],
                params['meta_learner.features.3.conv3.bias'],
                padding=1)
            out = F.batch_norm(
                out,
                params['meta_learner.features.3.bn3.running_mean'],
                params['meta_learner.features.3.bn3.running_var'],
                params['meta_learner.features.3.bn3.weight'],
                params['meta_learner.features.3.bn3.bias'],
                momentum=1,
                training=True)
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, MP_SIZE)

            out = out.view(out.size(0), -1)
            out = F.linear(out, params['meta_learner.fc.weight'],
                           params['meta_learner.fc.bias'])

        out = F.log_softmax(out, dim=1)
        return out


def conv_block(index,
               in_channels,
               out_channels=N_FILTERS,
               padding=0,
               pooling=True):
    """
    The unit architecture (Convolutional Block; CB) used in the modules.
    The CB consists of following modules in the order:
        3x3 conv, 64 filters
        batch normalization
        ReLU
        MaxPool
    """
    if pooling:
        conv = th.nn.Sequential(
            OrderedDict([
                ('conv'+str(index), th.nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), th.nn.BatchNorm2d(out_channels, momentum=1, \
                    affine=True)),
                ('relu'+str(index), th.nn.ReLU(inplace=True)),
                ('pool'+str(index), th.nn.MaxPool2d(MP_SIZE))
            ]))
    else:
        conv = th.nn.Sequential(
            OrderedDict([
                ('conv'+str(index), th.nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), th.nn.BatchNorm2d(out_channels, momentum=1, \
                    affine=True)),
                ('relu'+str(index), th.nn.ReLU(inplace=True))
            ]))
    return conv
