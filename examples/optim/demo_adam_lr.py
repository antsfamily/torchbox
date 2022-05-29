import math
import torch as th
import matplotlib.pyplot as plt

lr = 1e-3
steps = 1000


params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
          th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
          }

betas = (0.9, 0.99)
optimizer = th.optim.Adam(params, lr=lr, betas=betas)
# optimizer = th.optim.SGD(params, lr=lr, momentum=0.9)
# optimizer = th.optim.Adadelta(params, lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
# optimizer = th.optim.Adagrad(params, lr=lr, lr_decay=0.3, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9988, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=None, epochs=num_epochs, steps_per_epoch=num_batch, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1)


lrs = []
for t in range(1, steps):
    optimizer.step()
    # print(optimizer.param_groups)
    lrs.append(lr * math.sqrt(1. - betas[1]**t) / (1. - betas[0]**t))


plt.figure()
plt.plot(lrs)
plt.show()
