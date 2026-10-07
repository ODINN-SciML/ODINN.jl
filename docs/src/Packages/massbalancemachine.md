# MassBalanceMachine.jl

[`MassBalanceMachine.jl`](https://github.com/ODINN-SciML/MassBalanceMachine.jl) is the data-driven surface mass balance module of the ODINN ecosystem. It ports neural network models trained with the Python [MassBalanceMachine](https://github.com/ODINN-SciML/MassBalanceMachine) into [`Lux.jl`](https://github.com/LuxDL/Lux.jl), so that they can be used as drop-in replacements for the temperature-index models of [`Muninn`](muninn.md).

The central type is `CustomMLP`, a subtype of `MBmodel` (via the intermediate `MLmodel` abstract type). Because it is an `MBmodel`, it plugs into exactly the same slot as `TImodel1` in a `Model`, and every downstream `Huginn` or `ODINN` workflow accepts it without modification. Machine learning mass balance models are the *de facto* data-driven surface mass balance option in the ecosystem — see Sjursen et al. [sjursen_machine_2025](@cite).

`MassBalanceMachine` re-exports all of `Muninn` (and therefore `Sleipnir`), so `using MassBalanceMachine` is enough to build glaciers and parameters alongside the model.

## Use directly vs. use `ODINN.jl`

Use `MassBalanceMachine` directly when you want to:

  - Load, inspect or register a pre-trained MassBalanceMachine model without running any ice flow simulation.
  - Evaluate a machine learning mass balance model on climate data, e.g. to compare it against a calibrated temperature-index model.
  - Convert a Python MassBalanceMachine export into a Julia-native `Lux.jl` network.

Use `Huginn` when you want a coupled ice dynamics + mass balance **forward** simulation driven by such a model, and `ODINN.jl` when you additionally need gradients through that coupling.

## Climate data requirements

Unlike the temperature-index models, `CustomMLP` requires **ERA5** forcing: it declares `Muninn.required_climate_data_source(::CustomMLP) = :ERA5` because the networks consume additional atmospheric variables (shortwave radiation `ssrd`, surface fluxes, …) that are not present in the default W5E5 files. Preprocessing those glaciers therefore requires running [`Gungnir`](gungnir.md) in ERA5 mode, which needs a CDS API key.

Models whose input features include `slope` or `aspect` additionally request dynamic topography, through `requires_dynamic_topography` and `mb_inputs`, so that the topographic fields are recomputed from the evolving surface as the simulation advances.

## Minimal usage example

```julia
using MassBalanceMachine

# Download a pre-trained model from the HuggingFace registry and store it locally
download_MLP("mlp_noSvf_wgms11_small_0.1")

# …or load a Python export directly, then register it for fast reuse
mlp = CustomMLP("path/to/params.json", "path/to/model.json")
save_model(mlp, "norway_nongeo")   # → ~/.MassBalanceMachine/models/
mlp = load_model("norway_nongeo")

list_models()   # show everything in the local registry
```

Once loaded, the model is used exactly like any other `MBmodel`:

```julia
using ODINN

model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = mlp
)
```

See the [Models page](../models.md#Mass-balance-models) for the full walkthrough, including the model registry API.

## Extending MassBalanceMachine

`MassBalanceMachine.jl` is itself an example of how to extend `Muninn`: it defines a new `MBmodel` subtype and implements the optional dispatch hooks (`compute_MB`, `requires_dynamic_topography`, `topography_window_m`, `mb_inputs`, `required_climate_data_source`). If you want to add a different architecture — or a different data-driven mass balance model altogether — see the [Extending ODINN](../extending.md#add-a-new-mass-balance-model) guide, and use `src/mass_balance_utils.jl` in the `MassBalanceMachine.jl` repository as the reference implementation.

Only multilayer perceptrons (MLPs) are covered so far. A lower-level `MLP(nNeurons, activation)` constructor is available if you want to build a network from scratch rather than importing one.

## API reference

See [MassBalanceMachine API](../API/api_massbalancemachine.md) for the full list of exported types and functions.

## Further reading

  - [Models](../models.md) — how mass balance models fit into a `Model`
  - [Muninn package page](muninn.md) — the temperature-index alternatives
  - [Gungnir package page](gungnir.md) — producing the ERA5 forcing these models require
  - [`MassBalanceMachine.jl` repository](https://github.com/ODINN-SciML/MassBalanceMachine.jl) — model training and the full registry API
