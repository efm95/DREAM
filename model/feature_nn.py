import torch
import torch.nn as nn

from model.gcu import *

class FeatureNN(torch.nn.Module):
    """Neural Network architecture for each individual feature."""

    def __init__(
        self,
        input_shape: int,
        feature_num: int,
        num_units: int,
        dropout: float,
        hidden_sizes: list = [64, 32]
    ) -> None:
        
        """
        Args:
            input_shape (integer): tensor input shape.
            feature_num: (integer): index of the feature nn.
            num_units (integer): number of hidden units in first hidden layer.
            dropout (float): percentage of dropout regularization.
            hidden sizes (list): list of sizes of each layer (except input layer and output layer)
        """
        super(FeatureNN, self).__init__()
        self._input_shape = input_shape
        self._num_units = num_units
        self._feature_num = feature_num
        self._hidden_sizes = hidden_sizes
        
        all_hidden_sizes = [self._num_units] + self._hidden_sizes

        layers = []

        self.dropout = nn.Dropout(p=dropout)
        self.gcos = gcos()
    
        #First layer
        layers.append(nn.Linear(in_features=input_shape,out_features=num_units))
        
        #Hidden layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(nn.Linear(in_features,out_features))        
         
        #Last layer
        layers.append(nn.Linear(in_features=all_hidden_sizes[-1], out_features=1, bias=False))
        
        self.model = nn.ModuleList(layers)
        
    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training mode."""
        
        outputs = inputs.unsqueeze(1)
        
        for layer in self.model[:-1]:
            outputs = self.dropout(self.gcos(layer(outputs)))
        outputs = self.model[-1](outputs)

        return outputs
    