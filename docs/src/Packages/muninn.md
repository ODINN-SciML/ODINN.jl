# Muninn.jl

[`Muninn.jl`](https://github.com/ODINN-SciML/Muninn.jl) is the surface mass balance (SMB) module of the ODINN ecosystem. It computes the net ice accumulation and ablation at the glacier surface — the key atmospheric forcing that drives glacier volume change over time.

The default model is a distributed temperature-index (TI) model, which approximates ablation from positive degree-days (PDD) and accumulation from solid precipitation. Two variants are provided: `TImodel1` (single degree-day factor, mutable — suitable for calibration) and `TImodel2` (separate snow and ice DDFs, immutable — suitable for forward simulation). Both are calibrated against geodetic mass balance observations from Hugonnet et al. 2021 [hugonnet_accelerated_2021](@cite) using `calibrate_ti_model!`.

Neural-network-based mass balance is supported via the [`MassBalanceMachine.jl`](https://github.com/ODINN-SciML/MassBalanceMachine.jl) extension, which ports pre-trained PyTorch models exported as JSON into `Lux.jl` as a `CustomMLP <: MBmodel`. See the [Models page](../models.md) for details.

`Muninn` re-exports all of `Sleipnir`, so importing `Muninn` gives access to the full Sleipnir API without a separate `using Sleipnir` statement.

## Use directly vs. use `ODINN.jl`

Use `Muninn` directly when you want to:

  - Compute or calibrate mass balance models independently of ice flow (e.g. sensitivity studies, regional mass balance assessments).
  - Plug in a custom SMB model into a downstream tool that accepts a `MBmodel` object.
  - Experiment with a new mass balance parameterization before integrating it into a full ODINN simulation.

Use `ODINN.jl` when you need the coupled ice dynamics + mass balance simulation, or when training a UDE that involves the mass balance component.

## Minimal usage example

```julia
using Muninn

params = Parameters(
    simulation = SimulationParameters(
    tspan = (2010.0, 2015.0),
    multiprocessing = false,
    use_MB = true
),
)

glaciers = initialize_glaciers(["RGI60-11.00897"], params)

# Calibrate a temperature-index model against geodetic observations
mb_models = [TImodel1(DDF = 6.0/1000.0) for _ in glaciers]
calibrate_ti_model!(mb_models, glaciers, params)

# Compute mass balance for one time step
glacier = glaciers[1]
step = params.simulation.step_MB  # monthly step (1/12 yr)
MB = MB_timestep(mb_models[1], glacier, step, 2010.5)
@show size(MB)   # (nx, ny) mass balance grid in m w.e.
```

## Extending Muninn

To add a new mass balance model (new `MBmodel` subtype), see the [Extending ODINN](../extending.md#add-a-new-mass-balance-model) guide.

## API reference

See [Muninn API](../API/api_muninn.md) for the full list of exported types and functions.
