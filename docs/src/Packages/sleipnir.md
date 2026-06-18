# Sleipnir.jl

[`Sleipnir.jl`](https://github.com/ODINN-SciML/Sleipnir.jl) is the foundational package of the ODINN ecosystem, providing the core data structures and infrastructure on which all other packages are built. Every other ODINN package (`Huginn`, `Muninn`, `ODINN`) depends on `Sleipnir`, and each re-exports its symbols so downstream users rarely need to import `Sleipnir` directly.

`Sleipnir` defines the glacier geometry and climate data containers (`Glacier2D`, `Climate2D`), the simulation parameter hierarchy (`Parameters`, `SimulationParameters`, `PhysicalParameters`), the law abstraction used to plug physical or machine-learning computations into the PDE solvers (`Law`, `AbstractLaw`), and the results container (`Results`). It also hosts the VJP infrastructure (`VJP`, `MatrixCache`, `ScalarCache`) used by inverse modelling workflows.

Data for `Sleipnir` is preprocessed by the Python package [`Gungnir`](gungnir.md) and stored under `~/.ODINN/ODINN_prepro/`. When preprocessing has been run, glacier objects are assembled with `initialize_glaciers()`, which reads the stored HDF5/JLD2 files. Pre-built datasets for common regions can be downloaded automatically without running `Gungnir` yourself.

## Use directly vs. use `ODINN.jl`

Use `Sleipnir` directly when you want to:

  - Build or inspect glacier data structures (`Glacier2D`, `Climate2D`) without running a full simulation.
  - Prototype a new `Law` type or `AbstractInput` that will later be used in `Huginn` or `ODINN`.
  - Write a lightweight script that reads preprocessed glacier data and extracts fields (thickness, surface elevation, climate) without loading the full simulation stack.

Use `ODINN.jl` when you need the end-to-end pipeline (forward simulation, calibration, UDE training, inversion) — it assembles `Sleipnir` types into runnable workflows for you.

## Minimal usage example

```julia
using Sleipnir

# Construct simulation parameters (multiprocessing disabled for local runs)
params = Parameters(
    simulation = SimulationParameters(
        tspan = (2010.0, 2015.0),
        multiprocessing = false,
        use_MB = true
    ),
    physical = PhysicalParameters()
)

# Load pre-initialized glaciers (requires preprocessed data in ~/.ODINN/)
glaciers = initialize_glaciers(["RGI60-11.00897"], params)

glacier = glaciers[1]
@show glacier.rgi_id, glacier.nx, glacier.ny
@show size(glacier.H₀)   # initial ice thickness grid
@show size(glacier.S)    # surface elevation grid
```

## How to extend Sleipnir

### Add a new Law (custom physics or ML computation)

A `Law` wraps a computation that is called at each ODE step (or at a fixed callback frequency). It can hold a cache, support custom VJPs for inverse modelling, and carry a name for dispatch.

```julia
using Sleipnir

# Define a custom diffusivity law: D = A * H^(n+2)
my_law_f! = function (cache, H, S, model, t, glacier_idx)
    @. cache.output = model.iceflow.A.f * H^4  # simplified SIA diffusivity
end

my_law = Law(
    f! = my_law_f!,
    init_cache = (model, glacier) -> MatrixCache(size(glacier.H₀)...),
    name = :MyCustomA
)
```

See the [Laws tutorial](../laws.md) and [Laws VJP tutorial](../vjp_laws.md) for full worked examples including VJP customization for AD compatibility.

### Add a new dynamic input type

Dynamic inputs are values recomputed each ODE step and passed to laws (e.g. surface slope, surface temperature). Subtype `AbstractInput` and implement `generate_inputs`.

See [Laws inputs tutorial](../input_laws.md) for the full pattern.

### Add new observation data to `Glacier2D`

`Glacier2D` accepts optional data fields (`ThicknessData`, `SurfaceVelocityData`, `DhdtData`) as parametric type parameters, using `Nothing` as the absent case. To add a new data type, follow the same pattern in `src/glaciers/data/`.

## API reference

See [Sleipnir API](../API/api_sleipnir.md) for the full list of exported types and functions.
