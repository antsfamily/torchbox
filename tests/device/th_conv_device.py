import os
import torch
import torch.nn as nn
import time


device = 'cuda:0'
device = 'cuda:1'

niters = 1000

print("Torch version: ", torch.__version__)
print("Torch CUDA version: ", torch.version.cuda)
print("CUDNN Version: ", torch.backends.cudnn.version())
print("GPU Device: ", torch.cuda.get_device_name(int(device[-1])))

def profile(model, x, benchmark, deterministic, nb_iters):
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

    # warmup
    for _ in range(10):
        out = model(x)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nb_iters):
        out = model(x)
    torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / nb_iters


model1 = nn.Sequential(
    nn.Conv1d(24, 256, kernel_size=(12,), stride=(6,), groups=4),
    nn.ReLU(),
    nn.Conv1d(256, 256, kernel_size=(6,), stride=(3,), padding=(2,), groups=4),
    nn.ReLU(),
    nn.Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,), groups=4),
    nn.ReLU(),
)

model1.to(device=device)

x = torch.randn(64, 24, 224, device=device)

time0 = profile(model1, x, benchmark=False, deterministic=False, nb_iters=niters)
print('Conv1d model, benchmark=False, deterministic=False, {:.3f}ms/iter'.format(time0*1000))
time1 = profile(model1, x, benchmark=True, deterministic=False, nb_iters=niters)
print('Conv1d model, benchmark=True, deterministic=False, {:.3f}ms/iter'.format(time1*1000))
time2 = profile(model1, x, benchmark=False, deterministic=True, nb_iters=niters)
print('Conv1d model, benchmark=False, deterministic=True, {:.3f}ms/iter'.format(time2*1000))
time3 = profile(model1, x, benchmark=True, deterministic=True, nb_iters=niters)
print('Conv1d model, benchmark=True, deterministic=True, {:.3f}ms/iter'.format(time3*1000))

model2 = nn.Sequential(
    nn.Conv2d(8, 32, kernel_size=(8, 8), stride=(4, 4)),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU()
)
model2.to(device=device)

x = torch.randn(64, 8, 224, 224, device=device)
time0 = profile(model2, x, benchmark=False, deterministic=False, nb_iters=niters)
print('Conv2d model, benchmark=False, deterministic=False, {:.3f}ms/iter'.format(time0*1000))
time1 = profile(model2, x, benchmark=True, deterministic=False, nb_iters=niters)
print('Conv2d model, benchmark=True, deterministic=False, {:.3f}ms/iter'.format(time1*1000))
time2 = profile(model2, x, benchmark=False, deterministic=True, nb_iters=niters)
print('Conv2d model, benchmark=False, deterministic=True, {:.3f}ms/iter'.format(time2*1000))
time3 = profile(model2, x, benchmark=True, deterministic=True, nb_iters=niters)
print('Conv2d model, benchmark=True, deterministic=True, {:.3f}ms/iter'.format(time3*1000))

model3 = nn.Sequential(
    nn.Conv2d(8, 32, kernel_size=(8, 1), stride=(4, 1)),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 1)),
    nn.ReLU()
)
model3.to(device=device)

x = torch.randn(64, 8, 224, 224, device=device)
time0 = profile(model3, x, benchmark=False, deterministic=False, nb_iters=niters)
print('Conv2d1 model, benchmark=False, deterministic=False, {:.3f}ms/iter'.format(time0*1000))
time1 = profile(model3, x, benchmark=True, deterministic=False, nb_iters=niters)
print('Conv2d1 model, benchmark=True, deterministic=False, {:.3f}ms/iter'.format(time1*1000))
time2 = profile(model3, x, benchmark=False, deterministic=True, nb_iters=niters)
print('Conv2d1 model, benchmark=False, deterministic=True, {:.3f}ms/iter'.format(time2*1000))
time3 = profile(model3, x, benchmark=True, deterministic=True, nb_iters=niters)
print('Conv2d1 model, benchmark=True, deterministic=True, {:.3f}ms/iter'.format(time3*1000))

