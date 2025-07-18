# Glaciers

## Glacier types

Glaciers in `ODINN.jl` are represented by a `Glacier` type. Each glacier has its related `Climate` type. Since `ODINN.jl` supports different types of simulations, we offer the possibility to work on 1D (i.e. flowline), 2D (e.g. SIA) or even 3D (not yet implemented, e.g. Full Stokes).

```@docs
Sleipnir.Glacier2D{F <: AbstractFloat, I <: Integer}
Sleipnir.Glacier2D()
```

## Climate

Every glacier has its associated climate, following the same spatial representation (e.g. 2D). These are also retrieved using OGGM, and different types of climate can be used. By default we provide W5E5, which is downscaled (for now using very simple methods) to the glacier grid. 

```@docs
Sleipnir.Climate2D
```

## Initializing glaciers and their climate

In order to create `Glacier` types with information of a given glacier for a simulation, one can initialize a list of glaciers based on RGI (Randolph Glacier Inventory) IDs. Before running this, make sure to have used `Gungnir` to download all the necessary data for those glaciers in local. [Here](https://github.com/ODINN-SciML/Gungnir/blob/main/notebooks/Example.ipynb) you will find a notebook showing how to do so. Even easier, you can just double check that these glaciers are already available on the ODINN server. The list of the already processed glaciers can be obtained with `get_rgi_paths()`.

```@docs
Sleipnir.initialize_glaciers
```