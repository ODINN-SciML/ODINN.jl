# Types and functions

In this page, we will go through the main types (i.e. `struct`s) used in `ODINN.jl`'s architecture, and the main functions linked to those types.

## Parameters

There are different types of parameters, holding specific information for different modelling aspects. All the types of parameters are wrapped into a `Parameter` type, which is threaded throughout `ODINN.jl`. 

```@docs
Sleipnir.Parameters
ODINN.Parameters
```

The main types of parameters are the following ones:

### Simulation parameters

Simulation parameters are used to specify anything related to ODINN simulations, ranging from types, working directories to multiprocessing.

```@docs
Sleipnir.SimulationParameters
Sleipnir.SimulationParameters()
```

### Physical parameters

Physical parameters are used to store physical constants used in the physical and machine learning models. 

```@docs 
Sleipnir.PhysicalParameters
Sleipnir.PhysicalParameters()
```

### Physical parameters

Solver parameters determine all aspects related to the numerical scheme used to solve the differential equations of glacier ice flow.

```@docs
Huginn.SolverParameters
Huginn.SolverParameters()
```

### Hyperparameters

Hyperparameters determine different aspects of a given machine learning model. For now, these are focused on neural networks, but we plan to extend them in the future for other types of regressors. 

```@docs
ODINN.Hyperparameters
ODINN.Hyperparameters()
```

### UDE parameters

Universal Differential Equation (UDE) parameters are used to determine different modelling choices regarding the use of UDEs, such as wich sensitivity algorithm to use, which target (e.g. SIA parameter to target), or which optimization method to use.

```@docs 
ODINN.UDEparameters
ODINN.UDEparameters()
```

## Glaciers

Glaciers in `ODINN.jl` are represented by a `Glacier` type. Each glacier has its related `Climate` type. Since `ODINN.jl` supports different types of simulations, we offer the possibility to work on 1D (i.e. flowline), 2D (e.g. SIA) or even 3D (not yet implemented, e.g. Full Stokes).

```@docs
Sleipnir.Glacier2D
Sleipnir.Glacier2D()
```

Every glacier has its associated climate, following the same spatial representation (e.g. 2D):

```@docs
Sleipnir.Climate2D
``` 

In order to create `Glacier` types with information of a given glacier for a simulation, one can initialize a list of glaciers based on RGI (Randolph Glacier Inventory) IDs. Before running this, make sure to have used `Gungnir` to download all the necessary data for those glaciers, or double check that these glaciers are already available on the ODINN server. 

```@docs
Sleipnir.initialize_glaciers
```

## Models

There are 3 main types of models in `ODINN.jl`, iceflow models, mass balance models and machine learning models. These three families are determined by abstract types, with specific types being declared as subtypes of these abstract types to ensure compatibility through the ODINN ecosystem. 

The three main types of models are gathered in a type `Model` in the following way:

```@docs
ODINN.Model
ODINN.Model()
```
### Ice flow models

Ice flow models are used to solve the PDEs describing the gravitational flow of glaciers. All ice flow models must be a subtype of abstract type `IceflowModel`. Ice flow models are managed by `Huginn.jl`. 

The main type of ice flow model used in `ODINN.jl` right now is a 2D Shallow Ice Approximation (SIA) model *(Hutter, 1983)*. This is declared in the following way:

```@docs
Huginn.SIA2Dmodel
Huginn.SIA2Dmodel(params::Sleipnir.Parameters)
```

When a simulation will be run in `ODINN.jl` using an ice flow model, its related equation will be solved using `OrdinaryDiffEq.jl`. The related equation to a `SIA2Dmodel` is declared in its related util functions. Generally, these equations need to exist both in-place (to reduce memory allocations and ensure maximum performance, see example below) or out-of-place (to be more AD-friendly).

```@docs
Huginn.SIA2D!
```

### Mass balance models

Mass balance models are used to simulate the simplified thermodynamics of the forcing of the atmosphere on glaciers. As per ice flow models, all specific mass balance models needs to be a subtype of the abstract type `MBmodel`. Mass balance models are managed by `Muninn.jl`. For now, we have simple temperature-index models, with either one or two degree-day factors (DDFs) *(Hock, 2003)*:

```@docs
Muninn.TImodel1
Muninn.TImodel1(params::Sleipnir.Parameters)
```

Surface mass balance models are run in `DiscreteCallback`s from `OrdinaryDiffEq.jl`, which enable the safe execution during the solving of a PDE in specificly prescribed time steps determined in the `steps`field in [`Sleipnir.SimulationParameters`](@ref).

We soon plan to add compatibility with neural networks coming from the [MassBalanceMachine](https://github.com/ODINN-SciML/MassBalanceMachine), which should become the *de facto* surface mass balance model in the `ODINN.jl` ecosystem. 

### Machine Learning models

Machine learning models are used in the context of Universal Differential Equations (UDEs, *Rackauckas et al., 2020*) to parametrize or learn specific parts of differential equations. Machine Learning models are manage by `ODINN.jl`. As per the other types of models, all machine learning models need to be a subtype of the abstract type `MLmodel`. The default solution here is to use a neural network:

```@docs
ODINN.NN
ODINN.NN(params::Sleipnir.Parameters)
```

## Simulations

One can run different types of simulations in `ODINN.jl`. Any specific type of simulation must be a subtype of `Simulation`. All simulations share the same common interface designed around multiple dispatch. Basically, once a simulation type has been created, one can easily run by calling `run!(simulation)`.

The main types of simulations are the following ones:

### Prediction

A prediction, also known as a forward simulation, is just a forward simulation given a model configuration, based on parameters, glaciers and models. These are managed in `Huginn.jl`, since they do not involve any inverse methods nor parameter optimization.

```@docs
Huginn.Prediction
Huginn.Prediction(model::Sleipnir.Model, glaciers::Vector{G}, parameters::Sleipnir.Parameters) where {G <: Sleipnir.AbstractGlacier}
```

### Inversion

An inversion optimises a given set of model parameters, based on a given target and an optimizer. These are handled by `ODINN.jl`. 

```@docs
ODINN.Inversion
ODINN.Inversion(
    model::Sleipnir.Model,
    glaciers::Vector{G},
    parameters::Sleipnir.Parameters
    ) where {G <: Sleipnir.AbstractGlacier}
```

### Functional inversion

A functional inversion is the inversion of the parameters of a regressor (e.g. a neural network), which parametrize a function that modulates a parameter or set of parameters in a given mechanistic model (e.g. the SIA).

```@docs
ODINN.FunctionalInversion
ODINN.FunctionalInversion(
    model::Sleipnir.Model,
    glaciers::Vector{G},
    parameters::Sleipnir.Parameters
    ) where {G <: Sleipnir.AbstractGlacier}
```