# Huginn.jl

[`Huginn.jl`](https://github.com/ODINN-SciML/Huginn.jl) is the ice flow dynamics module of the ODINN ecosystem. It implements numerical solvers for glacier ice flow PDEs, with the 2D Shallow Ice Approximation (SIA2D) as the primary model. Ice flow PDEs are integrated using [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl), giving access to a wide range of adaptive time-stepping solvers and making the forward model compatible with SciML's sensitivity and adjoint infrastructure.

The main entry point is the `Prediction` simulation container, which holds the ice flow model, a list of glaciers, simulation parameters, and pre-allocated cache arrays. Calling `run!(prediction)` solves the ice thickness PDE for every glacier in the list (in parallel if `multiprocessing=true` in the simulation parameters) and stores time series of thickness, surface elevation, velocity, and mass balance in a `Results` object.

SIA2D is implemented as a fully in-place ODE right-hand side: `SIA2D!(dH, H, sim, t, θ)`. All intermediate fields (diffusivity `D`, fluxes, staggered-grid averages) are stored in a `SIA2DCache` and reused across time steps, making the solver memory-efficient and AD-compatible via Enzyme.

## Use directly vs. use `ODINN.jl`

Use `Huginn` directly when you want to:

  - Run **forward ice flow simulations** without UDE training or inversion — e.g. projecting glacier evolution under a climate scenario.
  - Benchmark or validate a new ice flow law against an analytical solution (e.g. the Halfar solution) or a mass conservation test.
  - Build a custom downstream tool that wraps the `Prediction` workflow without the full ODINN.jl training stack.

Use `ODINN.jl` when you need automatic differentiation through the ice flow solver, UDE training, or classical/functional inversion — `ODINN` wraps `Huginn`'s forward solver and adds the gradient/adjoint infrastructure.

## Minimal usage example

```julia
using Huginn, Sleipnir

params = Parameters(
    simulation = SimulationParameters(
    tspan = (2010.0, 2015.0),
    multiprocessing = false,
    use_MB = true
),
)

glaciers = initialize_glaciers(["RGI60-11.00897"], params)

# Build the ice flow model (SIA2D + temperature-index mass balance)
model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0)
)

# Run a forward simulation
prediction = Prediction(model, glaciers, params)
run!(prediction)

# Retrieve results
results = prediction.results[1]
@show results.H[end]   # final ice thickness
```

See the [Forward simulation tutorial](../forward_simulation.md) for a full worked example.

## The SIA2D execution chain

Understanding how SIA2D connects to OrdinaryDiffEq.jl is useful context for users and essential for developers writing a new model. The call chain is:

```
run!(prediction)
  └── batch_iceflow_PDE!(glacier_idx, simulation)        # dispatch point — override this for a new model
        ├── init_cache(model, ...)                       # allocate model cache
        ├── build_callback(model, cache, ...)            # law-update callbacks
        └── simulate_iceflow_PDE!(sim, cb, SIA2D_PDE!, tstops)
              └── ODEProblem(SIA2D_PDE!, H₀, tspan, simulation)
                    └── SIA2D_PDE!(dH, H, simulation, t)  # called at each ODE step
                          └── SIA2D!(dH, H, sim, t, θ)    # actual PDE kernel
```

There are two distinct RHS functions: **`SIA2D!`** is the PDE kernel (signature `(dH, H, simulation, t, θ)`, keeps `θ` for AD); **`SIA2D_PDE!`** is a thin adapter that drops `θ` to match OrdinaryDiffEq's `f(du, u, p, t)` convention. Both live in `prediction_utils.jl` / `SIA2D_utils.jl`.

## Extending Huginn

To add a new iceflow model (e.g. SSA, DIVA) or a new iceflow law, see the [Extending ODINN](../extending.md) guide. It uses the execution chain above as a reference and walks through the required interface layer by layer (forward simulation → inversion → velocity diagnostics), with a complete SSA2Dmodel skeleton.

## API reference

See [Huginn API](../API/api_huginn.md) for the full list of exported types and functions.
