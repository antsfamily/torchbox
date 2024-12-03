import torch as th

print("Torch Version: ", th.__version__)
print("CUDA Version: ", th.version.cuda)
print("CUDNN Version: ", th.backends.cudnn.version())
print("CUDA available?: ", th.cuda.is_available())
print("CUDA device count: ", th.cuda.device_count())

for gpuid in range(th.cuda.device_count()):
    print("GPU ID: %d, name: %s, capability: %s" % (gpuid, th.cuda.get_device_name(gpuid), th.cuda.get_device_capability(gpuid)))
    print(th.cuda.get_device_properties(gpuid))