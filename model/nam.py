import torch
import torch.nn as nn

from typing import Sequence
from typing import Tuple

from model.feature_nn import *

class NAM(torch.nn.Module):
    """Neural Additive Model (NAM)
    
    Create a nn model by generating a list of independent NN (using feature_nn). 
    """

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
        super(NAM, self).__init__()
        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        
        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

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


    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out)
        
        return dropout_out
        
        