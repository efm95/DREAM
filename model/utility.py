import torch

def device_identifier():
    #try: #torch 2.0 currently under development (08/02/2023)
    #    if torch.has_mps:
    #        dev = 'mps'
    #except:
    if torch.has_cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'
    return dev