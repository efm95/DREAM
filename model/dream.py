import numpy as np
import torch
import torch.nn as nn

from model.nam import *
from model.utility import * 

import logging
logging.basicConfig(format='%(asctime)s [%(filename)s] %(message)s',
    datefmt='%Y-%m-%d-%H:%M:%S',
    level=logging.INFO,
    encoding="utf-8")

class REM_NAM_logit(torch.nn.Module):
    """REM using NAM architecture and logit approximation"""
    
    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float
    ) -> None:
        """
        Args:
            num_inputs (int): number of inputs (covariates).
            num_units (list): list containing the size of each first layer, for each feature.
            hidden_sizes (list): lisst containing the size of each layer (same for each layer for simplicity).
            dropout (float): percentage of dropout on the output layer of each feature nn. 
            feature_dropout (float): percentage of dropout on each layer of each feature nn.
        """
        
        super(REM_NAM_logit,self).__init__()
        self.NAM = NAM(
            num_inputs=num_inputs,
            num_units = num_units,
            hidden_sizes = hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout
        )
        
        
    def forward(self,
                events:torch.tensor,
                non_events:torch.tensor):
        """
        Args:
            events (torch.tensor): event list with just covariates encoded as torch.float32.
            non_events (torch.tensor): non-event list with just covariates encoded as torch.float32.
        """
        
        out_events = self.NAM(events)
        out_n_events = self.NAM(non_events)
        out = out_events-out_n_events
        out = torch.sum(out,dim=-1)
        out = torch.sigmoid(out)
        return out


class DREAM:
    def __init__(self,
                 num_inputs:int,
                 first_layer:list,
                 single_layers:list,
                 dropout:float = 0.0,
                 feature_dropout:float=0.0) -> None:
        """DREAM initializer

        Args:
            num_inputs (int): number of covariates
            first_layer (list): list of first layer sizes (one for each covariate/input)
            single_layers (list): list of layer sizes (length of the list determines how deep the features are)
            dropout (float): percentage of dropout on the output of each feature. Defaults to 0.0.
            feature_dropout (float): percentage of dropout inside each feature. Defaults to 0.0.
        """
        
        self.num_inputs = num_inputs
        self.first_layer = first_layer
        self.single_layers = single_layers
        self.num_layers = len(single_layers)
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        
        self.device = device_identifier()
        
        self.model = REM_NAM_logit(num_inputs=self.num_inputs,
                             num_units=self.first_layer,
                             hidden_sizes=self.single_layers,
                             dropout=self.dropout,
                             feature_dropout=self.feature_dropout).to(device=self.device)
                
        
    def fit(self,
            events:torch.tensor,
            non_events:torch.tensor,
            epochs:int=100,
            batch_size:int=2**12,
            val_pos:int=0,
            lr:float=0.001,
            conv_tol:int=6,
            conv_jumps:int=15,
            verbose:bool=True,
            gradient_clipping:float = None):
        """
        Args:
            events (torch.tensor): event list with just covariates encoded as torch.float32.
            non_events (torch.tensor): non-event list with just covariates encoded as torch.float32.
            epochs (int): number of epochs. Defaults to 100.
            batch_size (int): size of the batches. Defaults to 2**12.
            val_pos (int): validation batch id. Defaults to 0.
            lr (float): learning rate. Defaults to 0.001.
            conv_tol (int): convergence tolerance based on how many digits should the rounding be after the comma on the loss. Defaults to 6.
            conv_jumps (int): convergence tolerance parameter, updates are stopped when no improvement of loss after "conv_jumps" steps. Defaults to 15.
            verbose (bool): updates verbose. Defaults to True.
            gradient_clipping (float): gradient clipping on the loss. If None, no gradient clipping on loss. Defaults to None.
        """
        
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
        stopping = False
        for epoch in range(epochs):
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
                loss = -y_hat.log().sum()/N #negative partial likelihood
                loss.backward()
                
                if gradient_clipping is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=gradient_clipping)

                self.optimizer.step()
            
                self.model.eval()
                with torch.no_grad():
                    y_val = self.model(event_val,non_event_val)
                    val_loss = -y_val.log().sum()/N_val
                    val_loss = val_loss.to(device='cpu')
            
                if verbose:
                    logging.info(f'Epoch {epoch+1} | Batch: {batch+1} | NLL: {loss.item()} | Val Loss: {val_loss.item()}')
            
                if np.round(val_loss.detach().numpy().item(),conv_tol) <= np.round(best_val_loss,conv_tol):
                    best_val_loss = val_loss.detach().numpy().item()
                    best_batch += 1
                else:
                    if (curr_batch-best_batch)==conv_jumps:
                        logging.info(f'Iteration stopped at epoch {epoch+1}, batch {batch+1}/{tot_batches+1} with NLL {loss.item()} and validation NLL {val_loss.item()}')
                        stopping = True
                        break
                
                if val_loss.item()<=0.001:
                    break
            if stopping == True:
                break
                            
    def feature_out(self,
                    nn_id:int,
                    input:torch.tensor):
        """Output of a single feature output

        Args:
            nn_id (int): index of neural network feature
            input (torch.tensor): input value

        Returns:
            covariate effect (torch.tensor): covariate effect for given input
        """
        
        self.model.eval()
            
        with torch.no_grad():
            return self.model.NAM.feature_nns[nn_id](input.to(device=self.device)).squeeze(1).detach()
        
    def fitted_values(self,
                      events:torch.tensor,
                      non_events:torch.tensor):
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(events,non_events).to(device='cpu')
        return out