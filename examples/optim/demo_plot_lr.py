import torch as th
import torchbox as tb
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

lr, num_epochs, batch_size, num_batch = 5e-3, 500, 8, 750
# lr, num_epochs, batch_size, num_batch = 1e-2, 1000, 8, 750

params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
          th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
          }

# optimizer = th.optim.Adam(params, lr=lr)
optimizer = th.optim.SGD(params, lr=lr, momentum=0.9)
# optimizer = th.optim.Adadelta(params, lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
# optimizer = th.optim.Adagrad(params, lr=lr, lr_decay=0.3, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
# scheduler = tb.optim.lr_scheduler.GaussianLR(optimizer, t_eta_max=120, sigma1=30, sigma2=100, eta_start=1e-5, eta_stop=1e-5, last_epoch=-1)
# scheduler = tb.optim.lr_scheduler.GaussianLR(optimizer, t_eta_max=80, sigma1=15, sigma2=200, eta_start=1e-4, eta_stop=1e-5, last_epoch=-1)
scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.78, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6, last_epoch=-1)
#scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=None, epochs=num_epochs, steps_per_epoch=num_batch, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1000.0, last_epoch=-1)

print(optimizer)
# print(scheduler)

lrs = []
for n in range(num_epochs):

    for b in range(num_batch):
        optimizer.step()
        # print(optimizer.param_groups)
        # lrs.append(optimizer.param_groups[0]['lr'])
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()
    # print(optimizer)

fontsize = 12
fonttype = 'Times New Roman'
fontdict = {'family': fonttype, 'size': fontsize}
plt.figure(figsize=(3.5, 3.5))
plt.grid()
plt.plot(lrs)
plt.xlabel('Epoch', fontdict=fontdict)
plt.ylabel('Learning rate', fontdict=fontdict)
plt.xticks(fontproperties=fonttype, size=fontsize)
plt.yticks(fontproperties=fonttype, size=fontsize)
plt.subplots_adjust(left=0.21, bottom=0.14, right=0.985, top=0.99, wspace=0.19, hspace=0.25)
plt.show()
