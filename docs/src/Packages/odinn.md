# ODINN.jl

[`ODINN.jl`](https://github.com/ODINN-SciML/ODINN.jl) is the top-level package of the ecosystem. It ties together the ice flow solver (`Huginn`), the mass balance models (`Muninn`), and the core data structures (`Sleipnir`) into an end-to-end differentiable glacier model. Its primary purpose is to enable **Universal Differential Equations (UDEs)**: hybrid models that combine physical PDEs with data-driven regressors such as neural networks, trained end-to-end with gradient-based optimization.

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
    UDE = UDEparameters(
        sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP()),
        optim_autoAD = Optimization.AutoEnzyme(),
        grad = ContinuousAdjoint()
    ),
    hyper = Hyperparameters(optimizer = Adam(0.01), epochs = 50)
)

glaciers = initialize_glaciers(["RGI60-11.00897", "RGI60-11.01450"], params)

# Build a UDE model: learn the creep coefficient A via a neural network
nn = NeuralNetwork(2 => [8, 8] => 1, params)
law_A = LawA(nn)
sia = SIA2Dmodel(params)
mb = TImodel1(DDF = 6.0/1000.0)
model = Model(sia, mb, TrainableComponents(A = law_A))

# Train on thickness observations
loss = LossH()
inv = Inversion(model, glaciers, params; loss)
run!(inv)   # calls train_UDE! internally
```

See the [Quick start](../quick_start.md) and [Functional inversion tutorial](../functional_inversion.md) for full worked examples.

## How to extend ODINN

### Add a new loss function

Subtype `AbstractLoss` and implement `loss()` and `backward_loss()`:

```julia
using ODINN

struct MyLoss <: AbstractLoss end

function ODINN.loss(::MyLoss, pred, obs, mask)
    # pred, obs: (nx, ny) arrays; mask: BitMatrix
    return mean((pred[.!mask] .- obs[.!mask]) .^ 2)
end

function ODINN.backward_loss(::MyLoss, pred, obs, mask)
    # return ∂L/∂pred
    return 2.0 .* (pred .- obs) .* .!mask ./ count(.!mask)
end
```

### Add a new inversion target

Subtype `AbstractSIA2DTarget` and implement `Diffusivity()` and the staggered-grid derivative methods. See `src/models/target/` for the existing `SIA2D_A_target`, `SIA2D_D_target`, and `SIA2D_D_hybrid_target` as templates.

### Add a new adjoint type

Subtype `AbstractAdjoint` in `src/inverse/AdjointTypes.jl` and implement the gradient computation. For manual adjoints through the SIA2D PDE, see `src/inverse/SIA2D/adjoint.jl`.

## API reference

See [ODINN API](../API/api_odinn.md) for the full list of exported types and functions.

## Further reading

  - [Functional inversion tutorial](../functional_inversion.md) — end-to-end UDE training
  - [Classical inversion tutorial](../classical_inversion.md) — gradient-free and gradient-based inversion
  - [Inversion types](../inversions.md) — overview of all inversion targets
  - [Optimization](../optimization.md) — optimizer configuration
  - [Sensitivity analysis](../sensitivity.md) — adjoint methods and `sensealg` configuration
  - [Laws tutorial](../laws.md) — building custom laws for use with `LawA`, `LawY`, `LawU`
