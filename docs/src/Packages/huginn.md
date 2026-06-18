# Huginn.jl

[`Huginn.jl`](https://github.com/ODINN-SciML/Huginn.jl) is the ice flow dynamics module of the ODINN ecosystem. It implements numerical solvers for glacier ice flow PDEs, with the 2D Shallow Ice Approximation (SIA2D) as the primary model. Ice flow PDEs are integrated using [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl), giving access to a wide range of adaptive time-stepping solvers and making the forward model compatible with SciML's sensitivity and adjoint infrastructure.

The main entry point is the `Prediction` simulation container, which holds the ice flow model, a list of glaciers, simulation parameters, and pre-allocated cache arrays. Calling `run!(prediction)` solves the ice thickness PDE for every glacier in the list (in parallel if `multiprocessing=true`) and stores time series of thickness, surface elevation, velocity, and mass balance in a `Results` object.

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

# Build the ice flow model with a temperature-dependent creep law
mb_model = TImodel1(DDF = 6.0/1000.0)
sia_model = SIA2Dmodel(params)
model = Model(sia_model, mb_model, nothing)

# Run a forward simulation
prediction = Prediction(model, glaciers, params)
run!(prediction)

# Retrieve results
results = prediction.results[1]
@show results.H[end]   # final ice thickness
```

See the [Forward simulation tutorial](../forward_simulation.md) for a full worked example.

## How to extend Huginn

### Add a new ice flow model

New ice flow models should subtype `IceflowModel` directly (defined in `Sleipnir`). `SIAmodel` is an intermediate abstract type specifically for Shallow Ice Approximation variants; models based on a different physical approximation — such as the Shallow Shelf Approximation (SSA) or DIVA — would sit at the `IceflowModel` level.

We plan to add new ice flow models to the ecosystem in the near future, including the SSA and the Depth-Integrated Viscosity Approximation (DIVA). The SSA is particularly relevant for fast-flowing outlet glaciers and ice streams where basal sliding dominates.

#### The SIA2D execution chain

Understanding how SIA2D connects to OrdinaryDiffEq.jl is essential before writing a new model. The call chain is:

```
run!(prediction)
  └── batch_iceflow_PDE!(glacier_idx, simulation)        # dispatch point — override this
        ├── init_cache(model, ...)                       # allocate model cache
        ├── build_callback(model, cache, ...)            # law-update callbacks
        └── simulate_iceflow_PDE!(sim, cb, SIA2D_PDE!, tstops)
              └── ODEProblem(SIA2D_PDE!, H₀, tspan, simulation)
                    └── SIA2D_PDE!(dH, H, simulation, t)  # called at each ODE step
                          └── SIA2D!(dH, H, sim, t, θ)    # actual PDE kernel
```

There are two distinct RHS functions: **`SIA2D!`** is the PDE kernel (signature `(dH, H, simulation, t, θ)`, keeps `θ` for AD); **`SIA2D_PDE!`** is a thin adapter that drops `θ` to match OrdinaryDiffEq's `f(du, u, p, t)` convention. Both live in `prediction_utils.jl` / `SIA2D_utils.jl` and are the analogues to implement for any new model.

#### Implementing SSA2Dmodel (example)

Five pieces are needed: the model struct, the cache struct, cache initialization, the PDE kernel, and the `batch_iceflow_PDE!` override that wires everything to the ODE solver.

!!! note

    The struct fields below are **illustrative only** — a real SSA implementation will require many more pre-allocated arrays (stress tensors, viscosity fields, staggered-grid buffers, law caches, etc.). Use [`SIA2Dmodel` and `SIA2DCache`](https://github.com/ODINN-SciML/Huginn.jl/blob/main/src/models/iceflow/SIA2D/SIA2D.jl) as the reference for the full set of fields needed, including law caches, VJP preparation, and mass balance buffers.

```julia
using Huginn, Sleipnir

# Type hierarchy:
#   IceflowModel (abstract, Sleipnir)
#     ├── SIAmodel (abstract) → SIA2Dmodel
#     └── SSA2Dmodel  ← new model directly under IceflowModel

# ── Model type: holds laws and configuration ──────────────────────────────
struct SSA2Dmodel <: IceflowModel
    viscosity_law::Any   # law for effective viscosity η
    friction_law::Any    # law for basal friction coefficient
    # ... (add all law fields required by the SSA kernel)
end

# ── Cache type: pre-allocated arrays, reused at every ODE step ────────────
# For a complete reference on what to pre-allocate, see SIA2DCache in SIA2D.jl
mutable struct SSA2DCache
    Ux::Matrix{Float64}   # x-velocity
    Uy::Matrix{Float64}   # y-velocity
    # ... (staggered-grid fields, stress tensors, law caches, MB fields, etc.)
    glacier_idx::Int
end

# ── Cache initializer: called once per glacier before the ODE solve ───────
function Sleipnir.init_cache(model::SSA2Dmodel, simulation, glacier_idx::Int, θ)
    g = simulation.glaciers[glacier_idx]
    nx, ny = g.nx, g.ny
    return SSA2DCache(zeros(nx, ny), zeros(nx, ny), glacier_idx)
end

# ── Callbacks: periodic law updates (return empty set if not needed) ──────
function Huginn.build_callback(model::SSA2Dmodel, cache::SSA2DCache,
        glacier_idx, θ, tspan)
    return CallbackSet()
end

# ── PDE kernel: analogous to SIA2D! — keep θ for AD compatibility ─────────
function SSA2D!(dU, U, simulation, t, θ)
    # write ∂U/∂t into dU using the SSA stress balance and mass continuity ...
end

# ── ODE adapter: drops θ to match ODEProblem's f(du,u,p,t) interface ─────
function SSA2D_PDE!(dU, U, simulation, t)
    SSA2D!(dU, U, simulation, t, nothing)
end

# ── Entry point: override batch_iceflow_PDE! to select SSA2D_PDE! ─────────
function Huginn.batch_iceflow_PDE!(glacier_idx::Int,
        simulation::Prediction{<:Sleipnir.Model{SSA2Dmodel}})
    params = simulation.parameters
    simulation.cache = Sleipnir.init_cache(simulation.model, simulation, glacier_idx, nothing)
    tstops = Huginn.define_callback_steps(params.simulation.tspan, params.solver.step)
    cb = build_callback(simulation.model.iceflow, simulation.cache.iceflow,
        glacier_idx, nothing, params.simulation.tspan)
    return Huginn.simulate_iceflow_PDE!(simulation, cb, SSA2D_PDE!, tstops)
end
```

### Add a new ice flow law

Laws in Huginn wrap parameterizations of the Glen flow law exponent `n`, sliding coefficient `C`, or creep coefficient `A`. Define a new `Law` subtype in `Sleipnir` (see [Sleipnir extension guide](sleipnir.md#how-to-extend-sleipnir)) and pass it to `SIA2Dmodel`.

See the [Laws tutorial](../laws.md) for a complete guide with callback-based and non-callback laws.

## API reference

See [Huginn API](../API/api_huginn.md) for the full list of exported types and functions.
