# Models

There are 3 main types of models in `ODINN.jl`, iceflow models, mass balance models and machine learning models. These three families are determined by abstract types, with specific types being declared as subtypes of these abstract types to ensure compatibility through the ODINN ecosystem.

The three main types of models are gathered in a type `Model` in the following way:

```@docs
ODINN.Model
ODINN.Model()
```
## Ice flow models

Ice flow models are used to solve the PDEs describing the gravitational flow of glaciers. All ice flow models must be a subtype of abstract type `IceflowModel`. Ice flow models are managed by `Huginn.jl`.

The main type of ice flow model used in `ODINN.jl` right now is a 2D Shallow Ice Approximation (SIA) model *(Hutter, 1983)*. This is declared in the following way:

```@docs
Huginn.SIA2Dmodel
```

When a simulation will be run in `ODINN.jl` using an ice flow model, its related equation will be solved using `OrdinaryDiffEq.jl`. The related equation to a `SIA2Dmodel` is declared in its related util functions. Generally, these equations need to exist both in-place (to reduce memory allocations and ensure maximum performance, see example below) or out-of-place (to be more AD-friendly).

```@docs
Huginn.SIA2D!
```

## Mass balance models

Mass balance models are used to simulate the simplified thermodynamics of the forcing of the atmosphere on glaciers. As per ice flow models, all specific mass balance models needs to be a subtype of the abstract type `MBmodel`. Mass balance models are managed by `Muninn.jl`. For now, we have simple temperature-index models, with either one or two degree-day factors (DDFs) *(Hock, 2003)*:

```@docs
Muninn.TImodel1
Muninn.TImodel1(params::Sleipnir.Parameters)
```

Surface mass balance models are run in `DiscreteCallback`s from `OrdinaryDiffEq.jl`, which enable the safe execution during the solving of a PDE in specificly prescribed time steps determined in the `steps`field in [`Sleipnir.SimulationParameters`](@ref).

We soon plan to add compatibility with neural networks coming from the [MassBalanceMachine](https://github.com/ODINN-SciML/MassBalanceMachine), which should become the *de facto* surface mass balance model in the `ODINN.jl` ecosystem.

## Machine Learning models

Machine learning models are used in the context of Universal Differential Equations (UDEs, *Rackauckas et al., 2020*) to parametrize or learn specific parts of differential equations. Machine Learning models are manage by `ODINN.jl`. As per the other types of models, all machine learning models need to be a subtype of the abstract type `MLmodel`. The default solution here is to use a neural network:

```@docs
ODINN.NeuralNetwork
```