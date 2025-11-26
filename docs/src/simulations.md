# Simulations

One can run different types of simulations in `ODINN.jl`. Any specific type of simulation must be a subtype of `Simulation`. All simulations share the same common interface designed around multiple dispatch. Basically, once a simulation type has been created, one can easily run by calling `run!(simulation)`.

The main types of simulations are the following ones:

## Prediction

A prediction, also known as a forward simulation, is just a forward simulation given a model configuration, based on parameters, glaciers and models. These are managed in `Huginn.jl`, since they do not involve any inverse methods nor parameter optimization.

```@docs
Huginn.Prediction
Huginn.Prediction(model::Sleipnir.Model, glaciers::Vector{G}, parameters::Sleipnir.Parameters) where {G <: Sleipnir.AbstractGlacier}
```

## Inversion

An `Inversion` is the inversion of the parameters involved in the PDE, or the parameters of a regressor (e.g. a neural network), which parametrize a function that modulates a parameter or set of parameters in a given mechanistic model (e.g. the SIA).

```@docs
ODINN.Inversion
ODINN.Inversion(
    model::M,
    glaciers::Vector{G},
    parameters::P
) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}
```