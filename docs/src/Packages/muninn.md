# Muninn.jl

[`Muninn.jl`](https://github.com/ODINN-SciML/Muninn.jl) is the surface mass balance (SMB) module of the ODINN ecosystem. It computes the net ice accumulation and ablation at the glacier surface — the key atmospheric forcing that drives glacier volume change over time.

The default model is a distributed temperature-index (TI) model, which approximates ablation from positive degree-days (PDD) and accumulation from solid precipitation. Two variants are provided: `TImodel1` (a single degree-day factor) and `TImodel2` (separate snow and ice degree-day factors).

Neural-network-based mass balance is supported via the [`MassBalanceMachine.jl`](https://github.com/ODINN-SciML/MassBalanceMachine.jl) extension, which ports pre-trained PyTorch models exported as JSON into `Lux.jl` as a `CustomMLP <: MBmodel`. See the [Models page](../models.md) for details.

`Muninn` re-exports all of `Sleipnir`, so importing `Muninn` gives access to the full Sleipnir API without a separate `using Sleipnir` statement.

## Use directly vs. use `ODINN.jl`

Use `Muninn` directly when you want to:

  - Compute mass balance independently of ice flow (e.g. sensitivity studies, regional mass balance assessments).
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

# Temperature-index mass balance model
glacier = glaciers[1]
mb_model = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0)

# Compute the mass balance for one monthly step, independently of ice flow
step = params.simulation.step_MB  # monthly step (1/12 yr)
get_cumulative_climate!(glacier.climate, 2010.5, step)
climate_2D = downscale_2D_climate(glacier.climate.climate_step, glacier.S, glacier.Coords)
MB = compute_MB(mb_model, climate_2D, step)
@show size(MB)   # (nx, ny) mass balance grid in m w.e.
```

## Extending Muninn

To add a new mass balance model (new `MBmodel` subtype), see the [Extending ODINN](../extending.md#add-a-new-mass-balance-model) guide.

## API reference

See [Muninn API](../API/api_muninn.md) for the full list of exported types and functions.
