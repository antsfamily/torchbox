
import torch as th


def device_transfer(obj, name, device):
    if name in ['optimizer', 'Optimizer']:
        for group in obj.param_groups:
            for p in group['params']:
                for k, v in obj.state[p].items():
                    if th.is_tensor(v):
                        obj.state[p][k] = v.to(device)
  
                    


