import torch as th
import torchvision
import torchvision.transforms as transforms
import torchbox as tb

# Device configuration
device = th.device('cuda:1' if th.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.01

lrfinder = tb.LrFinder(device)
# lrfinder = tb.LrFinder(plotdir='./')

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/mnt/d/DataSets/dgi/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/mnt/d/DataSets/dgi/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = th.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = th.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class ConvNet(th.nn.Module):  # Convolutional neural network (two convolutional layers)

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = th.nn.Sequential(
            th.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            th.nn.BatchNorm2d(16),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = th.nn.Sequential(
            th.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            th.nn.BatchNorm2d(32),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = th.nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss
criterion = th.nn.CrossEntropyLoss()

# optimizer
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)


lrfinder.find(train_loader, model, optimizer, criterion, nin=1,
              nbgc=1, lr_init=1e-8, lr_final=1e3, beta=0.98)

lrfinder.plot(lrmod='Linear')
lrfinder.plot(lrmod='Log')
