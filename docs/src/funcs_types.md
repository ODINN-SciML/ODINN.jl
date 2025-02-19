# Index of functions and types

## Parameters

There are different types of parameters, holding specific information for different modelling aspects. All the types of parameters are wrapped into a `Parameter` type, which is threaded throughout `ODINN.jl`. 

```@docs
Sleipnir.SimulationParameters
Sleipnir.PhysicalParameters
ODINN.Hyperparameters
ODINN.UDEparameters
```

All these subtypes of parameters are gathered in a `Parameters` struct:

```@docs
ODINN.Parameters
```

## Glaciers

Glaciers in `ODINN.jl` are represented by a `Glacier` type. Each glacier has its related `Climate`. Since `ODINN.jl` supports different types of simulations, we offer the possibility to work on 1D (i.e. flowline), 2D (e.g. SIA) or even 3D (not yet implemented, e.g. Full Stokes).

```@docs
Sleipnir.Glacier2D
```

Every glacier has its associated climate, following the same spatial representation (e.g. 2D):

<!-- ```@docs
Sleipnir.Climate2D
``` -->

## Models

There are 3 main types of models in `ODINN.jl`, iceflow models, mass balance models and machine learning models. 

Work in progress...