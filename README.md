# Deep Relational Event Additive Model

This repository contains the codes for the paper [Modelling non-linear Effects with Neural Networks in Relational Event Models](https://arxiv.org/abs/2312.12357)

## Abstract 
Dynamic networks offer an insight of how relational systems evolve. However, modeling these networks efficiently remains a challenge, primarily due to computational constraints, especially as the number of observed events grows. This project addresses this issue by introducing the Deep Relational Event Additive Model (DREAM) as a solution to the computational challenges presented by modeling non-linear effects in Relational Event Models (REMs). DREAM relies on Neural Additive Models to model non-linear effects, allowing each effect to be captured by an independent neural network. By strategically trading computational complexity for improved memory management and leveraging the computational capabilities of Graphic Processor Units (GPUs), DREAM efficiently captures complex non-linear relationships within data. This approach demonstrates the capability of DREAM in modeling dynamic networks and scaling to larger networks. Comparisons with traditional REM approaches showcase DREAM superior computational efficiency. The model potential is further demonstrated by an examination of the patent citation network, which contains nearly 8 million nodes and 100 million events.

## Use and train DREAM

Begin by cloning the repository and navigating to the DREAM directory:

```
git clone https://github.com/efm95/DREAM
cd DREAM
```

All model files are located in the model folder. You can import these models into your script as follows:

```
from model.dream import DREAM
from model.dream_gp import DREAM_gp
```

For detailed instructions on how to use the models, please refer to the provided Jupyter notebooks: `Tutorial_simulated_data.ipynb` and `Tutorial_data.ipynb`.
