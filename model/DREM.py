import numpy as np

from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from typing import Sequence
from typing import Tuple

from model.activation_functions import *
from utility import *

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d-%H:%M:%S',
    level=logging.INFO,
    encoding="utf-8")

class FeatureNN(torch.nn.Module):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        input_shape: int,
        feature_num: int,
        num_units: int,
        dropout: float,
        hidden_sizes: list = [64, 32],
        activation: str = 'exu'
    ) -> None:
        """Initializes FeatureNN hyperparameters.
        Args:
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          feature_num: Feature Index used for naming the hidden layers.
        """
        super(FeatureNN, self).__init__()
        self._input_shape = input_shape
        self._num_units = num_units
        self._feature_num = feature_num
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        
        all_hidden_sizes = [self._num_units] + self._hidden_sizes

        layers = []

        self.dropout = nn.Dropout(p=dropout)

        #self.tanh = nn.Tanh()
        self.cos = cos()
        #self.sin = sin()
        #self.rbf = rbf()
        #self.sin = sin()
        # #self.gelu = nn.GELU()
        # self.relu = nn.ReLU()
        #self.softmax = nn.Softmax()
        # #self.sigmoid = nn.Sigmoid()
        #self.silu = nn.SiLU()
        #self.sgelu = sgelu()
    
        # #First layer
        
        layers.append(nn.Linear(in_features=input_shape,out_features=num_units))
        #layers.append(cos())
        
        # #Hidden layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(nn.Linear(in_features,out_features))
            #layers.append(cos())
        
         #Last layer
        layers.append(nn.Linear(in_features=all_hidden_sizes[-1], out_features=1, bias=False))
        
        self.model = nn.ModuleList(layers)
        
        
        ###################
        ### PAPER MODEL ###
        ###################
        
        """
        # First layer is ExU
        if self._activation == "exu":
            layers.append(ExU(in_features=input_shape, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_shape, out_features=num_units))
        
        
        
        ## Hidden Layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(LinReLU(in_features, out_features))
            #layers.append(ExU(in_features,out_features))
            
        ## Last Linear Layer
        layers.append(nn.Linear(in_features=all_hidden_sizes[-1], out_features=1, bias=False))

        self.model = nn.ModuleList(layers)
        """
        
    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        outputs = inputs.unsqueeze(1)
        #outputs = self.tanh(self.model[0](outputs))
        
        for layer in self.model[:-1]:
            #outputs = self.dropout(layer(outputs))
            outputs = self.dropout(self.cos(layer(outputs)))
        outputs = self.model[-1](outputs)
            
        #outputs = self.model[-1](outputs)
        return outputs
    
class NAM(torch.nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float
    ) -> None:
        super(NAM, self).__init__()
        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        
        #self.final_linear = nn.Linear(in_features=num_inputs,out_features=1,bias=False)

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)
        #self.betas = nn.Parameter(torch.randn(int(self.num_inputs/2),1),requires_grad=True)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1, 
                num_units=self.num_units[i], 
                dropout=self.dropout, 
                feature_num=i, 
                hidden_sizes=self.hidden_sizes
            )
            for i in range(num_inputs)
        ])

        #self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out)
        
        #non_ev = torch.zeros([dropout_out.shape[0],int(dropout_out.shape[1]/2)])
        #ev = torch.zeros([dropout_out.shape[0],int(dropout_out.shape[1]/2)])
        #for i in range(non_ev.shape[1]):
        #    out[:,i] = torch.sub(dropout_out[:,(i*2)],dropout_out[:,(i*2)+1])
        #    non_ev[:,i] = torch.sum(dropout_out[:,i+1],dropout_out[:,i+2])
        #    ev[:,i] = torch.sum(dropout_out[:,(i*2)],dropout_out[:,(i*2)+1])
        
        #out = torch.matmul(out,self.betas)
        #out = torch.sum(dropout_out, dim=-1)
        #out = self.final_linear(dropout_out)
        
        #denom = torch.sum(dropout_out, dim=-1)
        
        #num = torch.sum(dropout_out[:,::2],dim=-1)
        
        return dropout_out
        
        
        
class REM_NAM(torch.nn.Module):
    
    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float
    ) -> None:
        
        super(REM_NAM,self).__init__()
        self.NAM = NAM(
            num_inputs=num_inputs,
            num_units = num_units,
            hidden_sizes = hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout
        )
        
        #self.final_linear = nn.Linear(in_features=num_inputs,out_features=1,bias=False)
        
    def forward(self,
                events:torch.Tensor,
                non_events:torch.Tensor):
        
        out_events = self.NAM(events)
        out_n_events = self.NAM(non_events)
        out = out_events-out_n_events
        #out = self.final_linear(out)
        out = torch.sum(out,dim=-1)
        out = torch.sigmoid(out)
        return out
        


class NeuREM:
    def __init__(self,
                 num_inputs:int,
                 first_layer:list,
                 single_layers:list,
                 dropout:float = 0.25,
                 feature_dropout:float=0.25) -> None:
        self.num_inputs = num_inputs
        self.first_layer = first_layer
        self.single_layers = single_layers
        self.num_layers = len(single_layers)
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        
        self.device = device_identifier()
        
        self.model = REM_NAM(num_inputs=self.num_inputs,
                             num_units=self.first_layer,
                             hidden_sizes=self.single_layers,
                             dropout=self.dropout,
                             feature_dropout=self.feature_dropout).to(device=self.device)
        
        #self.device = device_identifier()
        
        
    def fit(self,
            events:torch.tensor,
            non_events:torch.tensor,
            batch_size:int=2**12,
            val_pos:int=0,
            lr:float=0.001,
            l2_lambda:float= None,
            verbose:bool=True):
        
        events = torch.split(events,batch_size)
        non_events = torch.split(non_events,batch_size)
        
        tot_batches = len(events)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        
        event_val = events[val_pos].to(device=self.device)
        non_event_val = non_events[val_pos].to(device=self.device)
        N_val = len(event_val)
        
        best_val_loss = torch.tensor(float('Inf')).item()
        best_batch = 0
        curr_batch = 0
        
        for batch in range(tot_batches):
            if batch==val_pos:
                continue
            
            curr_batch+=1
            x_event = events[batch].to(device=device_identifier())
            x_non_event = non_events[batch].to(device=device_identifier())
            N = len(x_event)
            
            self.model.train()
            self.optimizer.zero_grad()
            
            y_hat = self.model(x_event,x_non_event)
            
            loss = -y_hat.log().sum()/N
            
            if l2_lambda is not None:
                l2_norm = sum(torch.linalg.norm(p, 2) for p in self.model.parameters())
                loss = loss+(l2_lambda * l2_norm)
                
            loss.backward()
            self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                y_val = self.model(event_val,non_event_val)
                val_loss = -y_val.log().sum()/N_val
                val_loss = val_loss.to(device='cpu')
            
            if verbose:
                logging.info(f'Batch: {batch+1} | NLL: {loss.item()} | Val Loss: {val_loss.item()}')
            
            if np.round(val_loss.detach().numpy().item(),4) <= np.round(best_val_loss,4):
                best_val_loss = val_loss.detach().numpy().item()
                best_batch += 1
            else:
                if (curr_batch-best_batch)==5:
                    logging.info(f'Iteration stopped at batch {batch+1}/{tot_batches+1} with NLL {loss.item()} and validation NLL {val_loss.item()}')
                    break
                
            if val_loss.item()<=0.001:
                break
                
    def feature_out(self,
                    nn_id:int,
                    input:torch.tensor):    
        self.model.eval()
        with torch.no_grad():
            return self.model.NAM.feature_nns[nn_id](input.to(device=self.device)).squeeze(1).detach()
        
