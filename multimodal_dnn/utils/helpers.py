import torch
import random
import string
import re

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_random_string(length):
    letters = string.ascii_lowercase + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def slugify(var):
    string = str(var)
    string = re.sub("[^0-9a-zA-Z]+", "-", string)
    string = string.strip('-')
    return string

def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)