# ODINN.jl

[`ODINN.jl`](https://github.com/ODINN-SciML/ODINN.jl) is the top-level package of the ecosystem. It ties together the ice flow solver (`Huginn`), the mass balance models (`Muninn`), and the core data structures (`Sleipnir`) into an end-to-end differentiable glacier model. Its primary purpose is to enable inverse modelling via **Universal Differential Equations (UDEs)**: hybrid models that combine physical PDEs with data-driven regressors.

`ODINN` provides:

  - **Classical inversion** — fit scalar or spatially distributed parameters (e.g. the Glen flow rate factor `A` or sliding coefficient `C`) to observed ice thickness or surface velocity, without neural networks.
  - **Functional inversion (UDE training)** — replace or augment a physical law with a neural network, train it with automatic differentiation through the PDE solver.
  - A **law system** (`LawA`, `LawY`, `LawU`) that wraps neural networks or custom functions into the SIA2D solver in a composable, AD-compatible way.
  - A **loss system** (`LossH`, `LossV`, `LossHV`, `MultiLoss`) for fitting to thickness and/or velocity observations.
  - **Sensitivity configuration** (`UDEparameters`, `sensealg`) exposing the full SciMLSensitivity adjoint zoo: continuous/discrete adjoints, Enzyme VJP, Mooncake, and manual adjoints.

## Use directly vs. using lower-level packages

Use `ODINN.jl` (i.e. `using ODINN`) when you need:

  - End-to-end UDE training or classical inversion.
  - Gradient-based optimization through the glacier PDE.
  - Access to the full `Inversion` workflow.

Use `Huginn` alone when you only need **forward simulation** (no gradients, no NN training). Use `Muninn` alone for mass balance computation. Use `Sleipnir` alone for data structure manipulation. This modular structure means each downstream use case only pays the compilation cost of what it needs.

## Minimal usage example

```julia
using ODINN

params = Parameters(
    simulation = SimulationParameters(
        tspan = (2010.0, 2015.0),
        multiprocessing = false,
        use_MB = true
    ),
    hyper = Hyperparameters(optimizer = ODINN.Adam(0.01), epochs = 50),
    UDE = UDEparameters(
        optim_autoAD = ODINN.NoAD(),
        grad = ContinuousAdjoint(),
        empirical_loss_function = LossH()   # fit to ice thickness
    )
)

glaciers = initialize_glaciers(["RGI60-11.00897", "RGI60-11.01450"], params)

# Build a UDE model: learn the creep coefficient A with a neural network.
# The NN-backed law is attached to the iceflow model, and the same NN is
# registered as the regressor for A.
nn_model = NeuralNetwork(params)
A_law = LawA(nn_model, params)
model = Model(
    iceflow = SIA2Dmodel(params; A = A_law),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
    regressors = (; A = nn_model)
)

# Train (the glaciers must carry the observations to fit to — see the
# functional inversion tutorial for generating/loading reference data)
functional_inversion = Inversion(model, glaciers, params)
run!(functional_inversion)
```

See the [Quick start](../quick_start.md) and [Functional inversion tutorial](../functional_inversion.md) for full worked examples.

## Extending ODINN

To add a new loss function, a new inversion target, or a new adjoint type, see the [Extending ODINN](../extending.md#add-a-new-loss-function) guide.

## API reference

See [ODINN API](../API/api_odinn.md) for the full list of exported types and functions.

## Further reading

  - [Functional inversion tutorial](../functional_inversion.md) — end-to-end UDE training
  - [Classical inversion tutorial](../classical_inversion.md) — gradient-free and gradient-based inversion
  - [Inversion types](../inversions.md) — overview of all inversion targets
  - [Optimization](../optimization.md) — optimizer configuration
  - [Sensitivity analysis](../sensitivity.md) — adjoint methods and `sensealg` configuration
  - [Laws tutorial](../laws.md) — building custom laws for use with `LawA`, `LawY`, `LawU`
