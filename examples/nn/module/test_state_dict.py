import torch as th


class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.W = th.nn.Parameter(th.randn(3, 3))

        self.layer = th.nn.Sequential(
            th.nn.Conv2d(2, 6, 3),
        )

        # self.conv1 = th.nn.Conv2d(2, 6, 3)

    def forward(self, x):

        return self.layer(x)
    
model = Net()

print(model.state_dict())
print(model.parameters())
for p in model.parameters():
    print(p.shape)

print(model.named_parameters())
for k, p in model.named_parameters():
    print(k, p.shape)