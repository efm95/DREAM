# Deep Relational Event Additive Model

This repository contains the codes for the paper [Modeling non-linear Effects with Neural Networks in Relational Event Models](https://arxiv.org/abs/2312.12357)

## Abstract 
Dynamic networks offer an insight of how relational systems evolve. However, modeling these networks efficiently remains a challenge, primarily due to computational constraints, especially as the number of observed events grows. This project addresses this issue by introducing the Deep Relational Event Additive Model (DREAM) as a solution to the computational challenges presented by modeling non-linear effects in Relational Event Models (REMs). DREAM relies on Neural Additive Models to model non-linear effects, allowing each effect to be captured by an independent neural network. By strategically trading computational complexity for improved memory management and leveraging the computational capabilities of Graphic Processor Units (GPUs), DREAM efficiently captures complex non-linear relationships within data. This approach demonstrates the capability of DREAM in modeling dynamic networks and scaling to larger networks. Comparisons with traditional REM approaches showcase DREAM superior computational efficiency. The model potential is further demonstrated by an examination of the patent citation network, which contains nearly 8 million nodes and 100 million events.

