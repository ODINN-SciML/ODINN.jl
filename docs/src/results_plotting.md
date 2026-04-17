# Results and plotting

## Results

Every `Simulation` type has an associated `Results` object(s), one for each one of the glaciers in the simulation. This object, as its name indicates, stores all the results of the simulation, which can be used for data analysis and plotting. These types are handled by `Sleipnir.jl`.

```@docs
Sleipnir.Results
Sleipnir.Results(glacier::G, ifm::IF) where {G <: AbstractGlacier, F <: AbstractFloat, IF <: AbstractModel, I <: Integer}
```

## Plots

One of the main things one can do with a `Results` object, is plotting them. The main function to do so is the following one:

```@docs
Sleipnir.plot_glacier
```

Another option is to generate a video of the evolution of the glacier's ice thickness during the simulation:

```@docs
Sleipnir.plot_glacier_vid
```

And finally, it is also possible to plot various gridded data on a glacier with the following function:

```@docs
Sleipnir.plot_gridded_data
```

It is also possible to accumulate gridded data over time and plot cumulative fields with the following functions:

```@docs
Sleipnir.accumulate_gridded_data
Sleipnir.plot_cumulative_gridded_data
Sleipnir.plot_cumulative_mb
```

For quick DEM visualizations from either a `Results` object or a glacier object, the following function is available:

```@docs
Sleipnir.plot_glacier_dem
```

And finally, figures can be saved with a unified utility function:

```@docs
Sleipnir.save_figure
```
