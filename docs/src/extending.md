# Extending ODINN

This page is the consolidated guide for developers who want to add new physics, models, or algorithms to the ODINN ecosystem. Each section corresponds to one extension point and describes the minimum interface to implement, plus pointers to deeper material.

## Add a new iceflow model

New iceflow models subtype `IceflowModel` (defined in `Sleipnir`). `SIAmodel` is an intermediate abstract type for Shallow Ice Approximation variants; a model based on a different physical approximation — Shallow Shelf Approximation, DIVA, etc. — sits directly under `IceflowModel`.

Before reading this section, it helps to understand how the existing SIA2D model is wired into OrdinaryDiffEq.jl — see the [execution chain diagram in the Huginn package page](Packages/huginn.md#the-sia2d-execution-chain).

The interface builds up in three layers depending on what you need:

**Layer 1 — Forward simulation** (run ice thickness evolution, no gradients):

| What you need to provide               | Why                                                                                                                                                                     |
|:-------------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `batch_iceflow_PDE!` override          | Entry point: tells Huginn which ODE right-hand side function to use for your model. Without this, Huginn falls back to `SIA2D_PDE!`.                                    |
| `init_cache(model::XYZmodel, ...)`     | Pre-allocates all the working arrays your physics needs (velocity fields, diffusivity, staggered-grid buffers, …). Called once per glacier before the ODE solve starts. |
| `cache_type(model::XYZmodel)`          | Returns the type of your cache struct. Needed for type-stable dispatch inside the solver.                                                                               |
| `build_callback(model::XYZmodel, ...)` | Builds any periodic callbacks (e.g. updating a law that changes with time). Return `CallbackSet()` if none are needed.                                                  |
| `apply_all_non_callback_laws!(...)`    | Applies your model's parametrized laws (e.g. Glen's A, sliding C) *inside* each ODE step. **Must implement** — the default throws.                                      |
| `apply_all_callback_laws!(...)`        | Applies the complementary laws *at callback frequency* (outside the ODE step). **Must implement** — the default throws.                                                 |

**Layer 2 — Inversion and adjoint differentiation** (needed if you want to use this model with `Inversion` or UDE training):

| What you need to provide                                        | Why                                                                                        |
|:--------------------------------------------------------------- |:------------------------------------------------------------------------------------------ |
| `precompute_all_VJPs_laws!(model, cache, sim::Prediction, ...)` | Forward-run stub — just `return nothing`.                                                  |
| `precompute_all_VJPs_laws!(model, cache, sim::Inversion, ...)`  | Real implementation in `ODINN.jl`: caches the law Jacobians before the adjoint solve runs. |

**Layer 3 — Surface velocity diagnostics** (optional):

| What you need to provide                | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:--------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `surface_V!` / `surface_V` / `V_from_H` | In the SIA the velocity is a **diagnostic** quantity: it is not part of the PDE state, so it has to be reconstructed from `H` and the surface gradient whenever velocities are needed for output or for a velocity loss. Models that solve a momentum balance directly — SSA, DIVA — carry the velocity in the PDE solution itself and do not need this reconstruction step. Implement these only if your model reconstructs velocity and does not inherit the SIA implementations in `SIA2D_utils.jl`. |

!!! note

    `apply_all_non_callback_laws!` and `apply_all_callback_laws!` have throwing generic fallbacks in `Sleipnir/src/laws/VJP.jl`. Missing them only surfaces as a runtime error when the law callback fires — not at construction time — so it can be easy to miss during early testing.

### Example skeleton — SSA2Dmodel

Five pieces are needed: the model struct, the cache struct, cache initialization, the PDE kernel, and the `batch_iceflow_PDE!` override.

!!! note

    The struct fields below are **illustrative only** — a real SSA implementation requires many more pre-allocated arrays (stress tensors, viscosity fields, staggered-grid buffers, law caches, etc.). See [`SIA2Dmodel` and `SIA2DCache`](https://github.com/ODINN-SciML/Huginn.jl/blob/main/src/models/iceflow/SIA2D/SIA2D.jl) for the full set of fields including law caches, VJP preparation, and mass balance buffers.

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

# ── Required: law application (non-callback laws applied inside ODE step) ─
function Huginn.apply_all_non_callback_laws!(model::SSA2Dmodel, cache::SSA2DCache,
        simulation, glacier_idx, t, θ)
    apply_law!(model.viscosity_law, cache.η, simulation, glacier_idx, t, θ)
    # ... apply all non-callback laws for this model
end

# ── Required: law application (callback laws applied at discrete steps) ───
function Huginn.apply_all_callback_laws!(model::SSA2Dmodel, cache::SSA2DCache,
        simulation, glacier_idx, t, θ)
    # apply laws that fire at callback frequency (e.g. sliding)
end

# ── Required for forward stub (inversion override lives in ODINN.jl) ──────
function Huginn.precompute_all_VJPs_laws!(model::SSA2Dmodel, cache::SSA2DCache,
        simulation::Prediction, glacier_idx, t, θ)
    nothing
end

# ── Callbacks: periodic law updates ───────────────────────────────────────
function Huginn.build_callback(model::SSA2Dmodel, cache::SSA2DCache,
        glacier_idx, tspan)
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

# ── Entry point: override batch_iceflow_PDE! to wire in SSA2D_PDE! ────────
function Huginn.batch_iceflow_PDE!(glacier_idx::Int,
        simulation::Prediction{<:Sleipnir.Model{SSA2Dmodel}})
    params = simulation.parameters
    simulation.cache = Sleipnir.init_cache(simulation.model, simulation, glacier_idx, nothing)
    tstops = Huginn.define_callback_steps(params.simulation.tspan, params.solver.step)
    cb = build_callback(simulation.model.iceflow, simulation.cache.iceflow,
        glacier_idx, params.simulation.tspan)
    return Huginn.simulate_iceflow_PDE!(simulation, cb, SSA2D_PDE!, tstops)
end
```

* * *

## Add a new mass balance model

New mass balance models subtype `MBmodel` (defined in `Muninn`). Mass balance is a source term of the ice flow right hand side,

```math
\frac{\partial H}{\partial t} = -\nabla\cdot(D\nabla S) + \dot m(H, t)
```

so a model supplies a **rate**, evaluated at every step of the solve — not an increment applied by a callback. That's what lets the automatic adjoint differentiate through mass balance, and it removes the operator-splitting error a periodic jump introduces.

Evaluating a model inside the right hand side isn't free: it runs far more often than a monthly callback did, can't read the climate rasters (`Rasters` isn't differentiable and isn't cheap), and must be differentiable in `H`. So climate is precomputed per mass balance window when the cache is built, and the model reads that.

```
Every RHS call:
  └── add_MB!(dH, H, simulation, t)
        └── MB_rate!(ṁ, H, mb_cache, mb_model, glacier, t)   ← implement this for your model
              (mb_cache carries the precomputed climate for the window containing t)

Once per simulation, per glacier:
  └── init_mb_cache(mb_model, simulation, glacier_idx, θ)    ← and this
```

**Minimum to implement:**

```julia
using Muninn

struct MyMBmodel <: MBmodel
    # your fields
end

# How the rate depends on the ice surface S, which decides how it can be evaluated.
# :elevation_only means it depends on S only through the scalar offset ΔS = S - ref_hgt,
# which is what makes a lookup table possible. Anything else is :general, the default.
Muninn.mb_S_dependence(::MyMBmodel) = :general

# Precompute whatever the RHS needs so that the solve never touches the climate rasters.
function Sleipnir.init_mb_cache(model::MyMBmodel, simulation, glacier_idx::Integer, θ)
    # return your cache type, or an empty one when use_MB is false
end

# The rate, in m of ice per year, written in place. Called at every step of the solve.
function Muninn.MB_rate!(ṁ, H, cache, model::MyMBmodel, glacier, t::Real)
    # ṁ[i, j] = ...
end

# Required to take gradients through the model with the manual adjoints. Free under
# SciMLSensitivityAdjoint, which differentiates MB_rate! itself.
function Muninn.MB_rate_∂H!(∂ṁ, H, cache, model::MyMBmodel, glacier, t::Real)
    # ∂ṁ[i, j] = ∂ṁ[i, j] / ∂H[i, j]   (diagonal: a cell depends only on its own H)
end
```

`compute_MB` is still required, and is unchanged:

```julia
function Muninn.compute_MB(model::MyMBmodel, climate_step::Climate2Dstep,
        step::AbstractFloat)
    # climate_step — gridded climate fields (temp, prcp, PDD, etc.)
    # step — fractional year length of this timestep
    # return a (nx, ny) matrix in m w.e.
end
```

It is what `calibrate_MB_model` fits parameters against, and what the test suite checks `MB_rate!` agrees with. If the two ever drift apart, the model being calibrated stops being the model being integrated, which nothing else guards against.

!!! note "Positivity is the model's responsibility"

    `ṁ` must vanish as `H` approaches zero, or a cell melts through the bed. `TImodel1` does this with a cubic `smoothstep` ramp on the rate, which is C¹ with a bounded derivative — a hard mask would be a step discontinuity in the state that no adaptive error controller can resolve and no adjoint can differentiate. See `mass_balance_rhs.jl` in Muninn.

!!! warning "Only `TImodel1` has a right hand side form today"

    `mb_S_dependence` defaults to `:general`, and the `:general` branch of `MB_rate!` is not implemented yet, so a model that does not opt into `:elevation_only` currently throws when the cache is built. Implementing it is tracked as its own piece of work.

**Optional dispatch hooks** (all have sensible defaults in Muninn — override only what differs):

```julia
Muninn.requires_dynamic_topography(::MyMBmodel) = false   # true if model uses slope/aspect
Muninn.topography_window_m(::MyMBmodel) = 200.0   # DEM smoothing radius (m)
Muninn.mb_inputs(::MyMBmodel) = (;)     # extra NamedTuple inputs
Muninn.required_climate_data_source(::MyMBmodel) = nothing # :ERA5 or :W5E5
Muninn.get_temp_bias(::MyMBmodel) = 0.0     # temperature offset (°C)
```

Pass your model to `Model(; iceflow = iceflow_model, mass_balance = MyMBmodel(...), regressors = ...)` as usual. Note that `Model` itself is defined in `Sleipnir`; `ODINN` only extends it (through `_construct_Model`) to build the `TrainableComponents` when `regressors` are provided.

!!! warning "TImodel2 is not yet fully implemented"

    `TImodel2` (separate snow/ice DDFs) is declared and exported in Muninn but has no `compute_MB` dispatch, and no right hand side form either. A simulation built with `TImodel2` fails when the mass balance cache is built. Full implementation is tracked in a separate Muninn issue.

* * *

## Add a new iceflow law

Laws are the primary mechanism for injecting custom or learnable physics into the iceflow solver. A `Law` wraps a computation — pure physics or a neural network — and is called at each ODE step (or at a fixed callback frequency).

**Where to add new law types:**

  - **Learnable law** (wraps a regressor, used in UDE training): add to [`ODINN.jl/src/laws/Laws.jl`](https://github.com/ODINN-SciML/ODINN.jl/blob/main/src/laws/Laws.jl)
  - **Non-learnable law** (pure physics, no neural network): add to [`Huginn.jl/src/laws/Laws.jl`](https://github.com/ODINN-SciML/Huginn.jl/blob/main/src/laws/Laws.jl)

**Minimal example** — a non-learnable diffusivity law:

```julia
using Sleipnir

struct MyDiffusivityLaw <: AbstractLaw{Matrix{Float64}}
    name::Symbol
    inputs::NamedTuple
    f!::Function
    init_cache::Function
    callback_freq::Union{Nothing, Real}
end

function MyDiffusivityLaw(; inputs = (;))
    MyDiffusivityLaw(
        :MyD,
        inputs,
        # f! receives (cache, inp, θ): inp is the NamedTuple of resolved inputs, θ holds NN params
        (cache,
            inp,
            θ) -> @. cache.output = inp.H ^ 3,
        (model, glacier) -> MatrixCache(glacier.nx, glacier.ny),
        nothing  # no callback; apply at every ODE step
    )
end
```

See the [Laws tutorial](laws.md) for complete worked examples (learnable and non-learnable), the [Laws inputs tutorial](input_laws.md) for implementing custom `AbstractInput` types, and the [Laws VJP tutorial](vjp_laws.md) for customizing adjoints for performance-sensitive laws.

For the conceptual overview of how `Law` binds inputs and a regressor to a target component, see the [Inversions page](inversions.md#understanding-the-laws-interface).

* * *

## Add a new loss function

A *loss function* measures the mismatch between the model's predicted state (ice thickness, surface velocity, etc.) and observations. For most of the losses, the metric itself is a **simple loss** (`AbstractSimpleLoss`, like the built-in `L2Sum` and `LogSum`); the composites `LossH`, `LossV`, `LossHV` (subtypes of `AbstractLoss`) then apply that metric to ice thickness and/or velocity. To add a new metric (e.g. a mean absolute error), subtype `AbstractSimpleLoss`. It needs a `distance` field — the composite uses it to build the in-glacier mask — and a `loss` method returning a scalar:

!!! note "Where the loss types live"

    The abstract types `GeneralAbstractLoss`, `AbstractSimpleLoss` and `AbstractLoss` are defined in `Sleipnir`, so that the velocity product needed by a loss can be determined from its type. The concrete losses (`L2Sum`, `LogSum`, `LossH`, `LossV`, `LossHV`, …) and the `loss`/`backward_loss` functions live in `ODINN`, which re-imports the abstract types. Subtyping `AbstractSimpleLoss` after `using ODINN` therefore works unchanged.

Beyond the per-timestep composites above, `ODINN` also provides **time-aggregated losses** (`LossDhdt`, `LossAvgV`, subtypes of `TimeAggregatedLoss`) which compare quantities integrated over the simulation window rather than pointwise in time, `MultiLoss` to combine several losses with weights, and a family of **regularization** terms (`TikhonovRegularization`, `InitialThicknessRegularization`, `VelocityRegularization`, `RheologyRegularization`, `DiffusivityRegularization`). Not all of these decompose into a simple-loss metric — `LossDhdt`, for instance, defines its own aggregation — so use them as templates when your new loss does not fit the `AbstractSimpleLoss` shape.

```julia
using ODINN

struct MyLoss <: AbstractSimpleLoss
    distance::Int
end
MyLoss(; distance = 3) = MyLoss(distance)

# a = prediction, b = reference (both (nx, ny)); mask is TRUE for valid in-glacier
# pixels; normalization is a scalar divisor. Must return a scalar.
function ODINN.loss(::MyLoss, a::Matrix, b::Matrix, mask::BitMatrix, normalization)
    return sum(abs.(a[mask] .- b[mask])) / normalization
end
```

Select it by wrapping it in a thickness/velocity loss and passing it through `UDEparameters` (there is no `loss` keyword on `Inversion`):

```julia
params = Parameters(
# …,
    UDE = UDEparameters(empirical_loss_function = LossH(loss = MyLoss()))   # LossV / LossHV for velocity
)
```

**Do you also need `backward_loss`?**

`backward_loss` returns `∂L/∂a` (same shape as `a`), zero outside the mask. It is only called by ODINN's manual adjoint methods (`DiscreteAdjoint` and `ContinuousAdjoint`). With `SciMLSensitivityAdjoint` (configured via `UDEparameters(grad = SciMLSensitivityAdjoint(), optim_autoAD = Optimization.AutoZygote())`), Zygote differentiates through the loss automatically and `backward_loss` is never called. You only need it for the manual adjoints:

```julia
function ODINN.backward_loss(::MyLoss, a::Matrix, b::Matrix, mask::BitMatrix, normalization)
    # ∂/∂a of sum|a − b|, restricted to valid pixels
    d = zero(a)
    d[mask] = sign.(a[mask] .- b[mask])
    return d ./ normalization
end
```

See [Sensitivity analysis](sensitivity.md) for a guide to choosing between adjoint methods.

!!! tip "Check your gradient numerically"

    After adding a loss (or an inversion target), verify the resulting gradient against finite differences with `grad_finite_diff(simulation)`, which returns the ratio, angle and relative error between the adjoint gradient and the finite-difference one. See [Numerical verification of the gradient](sensitivity.md#Numerical-verification-of-the-gradient).

## Add a new inversion target

**When do you need a custom target?** Only if you use ODINN's manual adjoint methods (`ContinuousAdjoint` or `DiscreteAdjoint`). Those methods require an explicit `AbstractSIA2DTarget` that hand-codes how your quantity enters the SIA2D diffusivity Jacobians. Currently implemented targets cover `A` (Glen flow rate factor) and `D` (diffusivity): `SIA2D_A_target`, `SIA2D_D_target`, `SIA2D_D_hybrid_target`.

If you use `SciMLSensitivityAdjoint` instead, no custom target is needed — Zygote + SciMLSensitivity differentiate through the full ODE automatically. This means parameters like the basal sliding coefficient `C` can already be inverted today via `SciMLSensitivityAdjoint`, simply by adding a `C` law to `SIA2Dmodel` and registering it in `TrainableComponents` — no new target code required.

**To add a manual adjoint target for a new quantity**, subtype `AbstractSIA2DTarget` and implement:

  - `Diffusivity(target; H̄, ∇S, θ, ...)` — the full diffusivity expression, including the contribution of the target quantity
  - `∂Diffusivity∂H`, `∂Diffusivity∂∇H`, `∂Diffusivity∂θ` — staggered-grid derivatives used by the adjoint
  - Optionally `Velocityꜛ` and its derivatives if you fit to surface velocity observations

The existing targets in [`src/models/target/`](https://github.com/ODINN-SciML/ODINN.jl/blob/main/src/models/target/) are the reference: `SIA2D_A_target` is the simplest (one scalar field), `SIA2D_D_hybrid_target` the most complex (combines A and D). Copy the nearest analogue and adapt the PDE terms.

!!! note

    Implementing a new target requires understanding how your quantity enters the diffusivity kernel — for `A` this is a straightforward linear factor, for `n` it involves `ln(H) · H^(n+2)` type terms (differentiating a power with respect to its exponent). If you are unsure, open a discussion on the [ODINN.jl issue tracker](https://github.com/ODINN-SciML/ODINN.jl/issues) — the maintainers are happy to help scope the work.
