# Glaciers

## Glacier types

Glaciers in `ODINN.jl` are represented by a `Glacier` type. Each glacier has its related `Climate` type. Since `ODINN.jl` supports different types of simulations, we offer the possibility to work on 1D (i.e. flowline), 2D (e.g. SIA) or even 3D (not yet implemented, e.g. Full Stokes). For now, all the simulations are workflows are focused on a 2D Shallow Ice Approximation (SIA).

```@docs
Sleipnir.Glacier2D{F <: AbstractFloat, I <: Integer}
Sleipnir.Glacier2D()
```

## Climate

Every glacier has its associated climate, following the same spatial representation (e.g. 2D). These are also retrieved using OGGM, and different types of climate can be used. By default we provide W5E5, which is downscaled (for now using very simple methods) to the glacier grid.

```@docs
Sleipnir.Climate2D
```

### Climate data

In ODINN we can leverage all the climate datasets available through OGGM. For more details, please check the [OGGM documentation regarding that](https://docs.oggm.org/en/stable/climate-data.html).

The main climate data supported include:

  - [W5E5](https://docs.oggm.org/en/stable/climate-data.html#w5e5)
  - [CRU](https://docs.oggm.org/en/stable/climate-data.html#cru)
  - [ERA5 and CERA-20C](https://docs.oggm.org/en/stable/climate-data.html#era5-and-cera-20c)
  - [HISTALP](https://docs.oggm.org/en/stable/climate-data.html#histalp)
  - Any other climate dataset. It is fairly easy to add climate datasets into OGGM.

## Initializing glaciers and their climate

### Standard workflow

Alpine glaciers are identified based on RGI (Randolph Glacier Inventory) IDs. There are two options to create `Glacier` types containing information about a given glacier for a simulation:

  - If the glacier is available in the preprocessed directory hosted on the [ODINN Hugging Face dataset](https://huggingface.co/datasets/ODINN-SciML/ODINN_prepro), there is nothing to do: ODINN downloads the required data automatically at precompilation. The list of already processed glaciers can be obtained with `get_rgi_paths()`.
  - If that is not the case, you have to use `Gungnir` to download all the necessary data for those glaciers locally. This package retrieves data from ERA5 (accessible through the Copernicus Climate Data Store) and from OGGM for the glacier outlines. [Here](https://github.com/ODINN-SciML/Gungnir/blob/main/notebooks/Example.ipynb) you will find a notebook showing how to do so. Once you have a local directory containing the data, you can override where ODINN looks for preprocessed directories by creating an `Overrides.toml` which should be placed in `~/.julia/artifacts/Overrides.toml`. It must contain the UUID of Sleipnir together with the path to your custom preprocessed directory:

```
[f5e6c550-199f-11ee-3608-394420200519]
ODINN_prepro = "/path/to/custom/dir"
```

Similarly, you can overwrite the Hugonnet et al. (2021) dataset if you want to use your own version:

```
[f5e6c550-199f-11ee-3608-394420200519]
hugonnet21_dataset = "/path/to/custom/dir"
```

See [the artifacts documentation](https://pkgdocs.julialang.org/v1/artifacts/) for more information.

```@docs
Sleipnir.initialize_glaciers
```

### Custom glaciers

Alternatively, users can create their own glaciers without using the automated initialization method [`initialize_glaciers`](@ref Sleipnir.initialize_glaciers) by manually specifying the attributes.
An empty glacier object without a grid can be created by using `Glacier2D()`.
When a grid is needed (most of the use-cases beyond debugging), each of fields have to be specified in [`Glacier2D()`](@ref Sleipnir.Glacier2D()).
