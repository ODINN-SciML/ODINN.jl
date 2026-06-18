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

| What you need to provide                | Why                                                                                                                                                         |
|:--------------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `surface_V!` / `surface_V` / `V_from_H` | Only needed if your model's velocity field is computed differently from SIA. The existing SIA implementations in `SIA2D_utils.jl` work for any SIA variant. |

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

# ── Entry point: override batch_iceflow_PDE! to wire in SSA2D_PDE! ────────
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

* * *

## Add a new mass balance model

New mass balance models subtype `MBmodel` (defined in `Muninn`). When an ODE step fires the MB callback, the application chain is:

```
MB_timestep!(glacier, cache, mb_model, t, step)
  └── compute_MB(mb_model, climate_2D_step, step)   ← implement this
        └── returns Matrix{Float} of MB values
  └── apply_MB_mask!(H, MB, MB_total)               ← handles ice-edge masking
        └── mutates H in-place; clips MB to prevent negative thickness
  └── pushes snapshot to cache.iceflow.MB_history
```

**Minimum to implement:**

```julia
using Muninn

struct MyMBmodel <: MBmodel
    # your fields
end

# Required: compute the distributed MB for one time step
function Muninn.compute_MB(model::MyMBmodel, climate_step::Climate2Dstep,
        step::AbstractFloat)
    # climate_step — gridded climate fields (temp, prcp, PDD, etc.)
    # step — fractional year length of this timestep
    # return a (nx, ny) matrix in m w.e.
end
```

**Optional dispatch hooks** (all have sensible defaults in Muninn — override only what differs):

```julia
Muninn.requires_dynamic_topography(::MyMBmodel) = false   # true if model uses slope/aspect
Muninn.topography_window_m(::MyMBmodel) = 200.0   # DEM smoothing radius (m)
Muninn.mb_inputs(::MyMBmodel) = (;)     # extra NamedTuple inputs
Muninn.required_climate_data_source(::MyMBmodel) = nothing # :ERA5 or :W5E5
Muninn.get_temp_bias(::MyMBmodel) = 0.0     # temperature offset (°C)
```

Pass your model to `Model(iceflow_model, MyMBmodel(...), trainable_components)` as usual.

!!! warning "TImodel2 is not yet fully implemented"

    `TImodel2` (separate snow/ice DDFs) is declared and exported in Muninn but has no `compute_MB` dispatch. A simulation built with `TImodel2` will fail at the first MB callback. Full implementation is tracked in a separate Muninn issue.

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

A *loss function* measures the mismatch between the model's predicted state (ice thickness, surface velocity, etc.) and observations. ODINN's built-in losses (`LossH`, `LossV`, `LossHV`) cover the most common cases. To add a different metric (e.g. an uncertainty-weighted MSE that uses the per-glacier observation uncertainties from Hugonnet et al. 2021) subtype `AbstractLoss` and implement `loss`, which must return a scalar:

```julia
using ODINN

struct MyLoss <: AbstractLoss end

function ODINN.loss(::MyLoss, pred, obs, mask)
    # pred, obs: (nx, ny) arrays of predicted/observed values (ice thickness, velocity, …)
    # mask: BitMatrix — true where data is MISSING; use .!mask to select valid pixels
    return mean((pred[.!mask] .- obs[.!mask]) .^ 2)
end
```

Pass it to an `Inversion` via the `loss` keyword: `Inversion(model, glaciers, params; loss = MyLoss())`.

**Do you also need `backward_loss`?**

`backward_loss` returns `∂L/∂pred` — the per-pixel gradient of the loss. It is only called by ODINN's manual adjoint methods (`DiscreteAdjoint` and `ContinuousAdjoint`). If you use `SciMLSensitivityAdjoint` (configured via `UDEparameters(grad = SciMLSensitivityAdjoint(), optim_autoAD = AutoZygote())`), Zygote differentiates through the loss automatically and `backward_loss` is never called. You only need it for manual adjoint configurations:

```julia
function ODINN.backward_loss(::MyLoss, pred, obs, mask)
    # ∂MSE/∂pred = 2*(pred - obs) / n_valid, zero where data is missing
    n = count(.!mask)
    return 2.0 .* (pred .- obs) .* .!mask ./ n
end
```

See [Sensitivity analysis](sensitivity.md) for a guide to choosing between adjoint methods.

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
