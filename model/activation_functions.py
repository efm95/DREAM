from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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
        output = F.tanh(output)#----> even better
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
        #output = F.elu(output)
        output = F.tanh(output)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
