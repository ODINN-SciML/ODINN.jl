# ODINN: toy version
OGGM (Open Global Glacier Model) + DIfferential equation Neural Networks

<img src="https://github.com/ODINN-SciML/odinn_toy/blob/main/plots/ODINN_toy.png" width="300">

Toy model with raw implementation of glacier mass balance and ice dynamics Universal Differential Equations (UDEs). 

It uses neural networks, differential equations and SINDy (Brunton et al., 2016) in order to combine mechanistic models describing glaciological processes (e.g. enhanced temperature-index model or the Shallow Ice Approximation) with machine learning. Neural networks are used to learn parts of the equations, which then can be interpreted in a mathematical form in order to update the original equation from the process. 
