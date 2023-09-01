import torch as th

th.manual_seed(2023)

class Net(th.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = th.nn.Conv2d(2, 1, 3, padding='same')

    def forward(self, x):

        return self.conv1(x)
    
model = Net()


x = th.rand(10, 2, 64, 64)
y = th.randn(10, 1, 64, 64)
x2 = th.rand(10, 2, 64, 64)

loss = th.nn.MSELoss()

print(0, model.conv1.weight.sum(), model.conv1.bias.sum())
print(0, model.conv1.weight.grad, model.conv1.bias.grad)
z = model(x)
z2 = model(x2)
print(1, model.conv1.weight.sum(), model.conv1.bias.sum())
print(1, model.conv1.weight.grad, model.conv1.bias.grad)
lossv = loss(z, y)
grads = th.autograd.grad(lossv, model.parameters(), create_graph=True)
print(2, model.conv1.weight.sum(), model.conv1.bias.sum())
print(2, model.conv1.weight.grad, model.conv1.bias.grad)
print('-------------------')
print(grads)
lossv.backward()
print(3, model.conv1.weight.sum(), model.conv1.bias.sum())
print(3, model.conv1.weight.grad, model.conv1.bias.grad)

