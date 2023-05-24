from turtle import shape
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class dSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,input):
        return torch.sigmoid(input)*(1+input*(1-torch.sigmoid(input)))

class cos(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.magnitude = torch.nn.Parameter(torch.ones(1))
        #self.freq = torch.nn.Parameter(torch.ones(1))
    def forward(self,input):
        return input*torch.cos(input=input)

class sin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,input):
        return input*torch.sin(input=input)
    
class rbf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,input):
        return torch.exp(-(1/2)*(input**2))
    

class sgelu(nn.Module):
    def __init__(self,slope=1) -> None:
        super().__init__()
        self.slope = slope
        
    def forward(self,input):
        return self.slope*input*torch.special.erf(input/np.sqrt(2))
    
    
class ExU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        
        self.sgelu=sgelu()
        #self.weights = Parameter(torch.rand(in_features,out_features))
        #self.bias = Parameter(torch.rand(in_features))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)
        #nn.init.xavier_uniform_(self.weights)
        #torch.nn.init.trunc_normal_(self.bias, std=0.5)
        

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
        #output = F.relu(output)
        #output = F.leaky_relu(output)
        #output = F.gelu(output)# ---> nice 
        #output = F.tanh(output)#----> even better
        #output = F.silu(output)
        output = self.sgelu(output)
        
        #output = F.leaky_relu(output)
        #output = torch.clamp(output, 0, n)
        
        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
    
class LinReLU(torch.nn.Module):
    __constants__ = ['bias']

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(LinReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.sgelu=sgelu()
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)
    
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = (inputs - self.bias) @ self.weights
        #output = F.relu(output)
        #output = F.gelu(output)
        #output = F.elu(output)
        #output = F.leaky_relu(output)
        #output = F.selu(output)
        #output = F.silu(output)
        #output = F.tanh(output)
        output = self.sgelu(output)
        
        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
