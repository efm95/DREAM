import torch
import torch.nn as nn

class gcos(nn.Module):
    """Compute Growig Cosine Unit (GCU) activation function"""
    def __init__(self) -> None:
        super().__init__()
    def forward(self,input):
        return input*torch.cos(input=input)