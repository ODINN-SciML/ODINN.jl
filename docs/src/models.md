# Models

There are 3 main types of models in `ODINN.jl`, iceflow models, mass balance models, and machine learning models. These three families are determined by abstract types, with specific types being declared as subtypes of these abstract types to ensure compatibility through the ODINN ecosystem.

The three main types of models are gathered in a parent type `Model` in the following way:

```@docs
ODINN.Model
ODINN.Model()
```

## Ice flow models

Ice flow models are used to solve the PDEs describing the gravitational flow of glaciers. All ice flow models must be a subtype of abstract type `IceflowModel`. Ice flow models are managed by the [`Huginn.jl`](https://github.com/ODINN-SciML/Huginn.jl) package.

The main type of ice flow model used in `ODINN.jl` right now is a 2D Shallow Ice Approximation (SIA) model *(Hutter, 1983)*. This is declared in the following way:

```@docs
Huginn.SIA2Dmodel
```

When a simulation will be run in `ODINN.jl` using an ice flow model, its related equation will be solved using `OrdinaryDiffEq.jl`. The related equation to a `SIA2Dmodel` is declared in its related util functions. These equations need to be defined in-place (to reduce memory allocations and ensure maximum performance, see example below). This is both compatible with the forward runs and with the reverse pass differentiated using `Enzyme.jl`.

```@docs
Huginn.SIA2D!
```

## Mass balance models

(Surface) Mass balance models are used to simulate the simplified thermodynamics of the forcing of the atmosphere on glaciers. As per ice flow models, all specific mass balance models needs to be a subtype of the abstract type `MBmodel`. Mass balance models are managed by [`Muninn.jl`](https://github.com/ODINN-SciML/Muninn.jl). For now, we have simple temperature-index models, with either one or two degree-day factors (DDFs) *(Hock, 2003)*:

```@docs
Muninn.TImodel1
Muninn.TImodel1(params::Sleipnir.Parameters)
```

Surface mass balance models are run in `DiscreteCallback`s from `OrdinaryDiffEq.jl`, which enable the safe execution during the solving of a PDE in specifically prescribed time steps determined in the `steps`field in [`Sleipnir.SimulationParameters`](@ref).

We soon plan to add compatibility with neural networks coming from the [MassBalanceMachine](https://github.com/ODINN-SciML/MassBalanceMachine), which should become the *de facto* surface mass balance model in the `ODINN.jl` ecosystem.

## Regressors

Regressors (e.g. machine learning models) are used in the context of Universal Differential Equations [rackauckas_universal_2020](@cite) to parametrize or learn specific parts of differential equations. Machine Learning models are managed by `ODINN.jl`. Virtually all available regressors in Julia can be used inside ODINN, but they need to be correctly interfaced. Here is an example of a simple neural network (multilayer perceptron) using `Lux.jl`:

```@docs
ODINN.NeuralNetwork
```

In order to parametrize a given variable inside an (ice flow) model, one can do it via the `regressors` keyword in `Model`:

```julia
nn_model = NeuralNetwork(params)
A_law = LawA(nn_model, params)
model = Model(
    iceflow = SIA2Dmodel(params; A = A_law),
    mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0),
    regressors = (; A = nn_model)
)
```
