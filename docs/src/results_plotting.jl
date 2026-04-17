# # Results and plotting tutorial

# This tutorial shows how to visualize glacier simulation results using the
# plotting utilities provided through ODINN.jl (from Sleipnir.jl).
# It covers the main plot types available after running a forward simulation.

using ODINN
using Sleipnir
using CairoMakie

# ## Running a forward simulation

# We first run a short forward simulation on Argentière glacier (RGI60-11.01450)
# over a 5-year period, which will serve as the basis for all the plots below.

rgi_ids = ["RGI60-11.01450"]
rgi_paths = get_rgi_paths()

params = Parameters(
    simulation = SimulationParameters(
        working_dir = joinpath(ODINN.root_dir, "demos"),
        tspan = (2010.0, 2015.0),
        multiprocessing = false,
        workers = 1,
        climate_data_source = :ERA5,
        rgi_paths = rgi_paths,
        use_MB = true,
        use_velocities = true,
        use_glathida_data = true,
        use_iceflow = true
    ),
    solver = SolverParameters(
        step = 1 / 12,
        progress = true,
        save_everystep = true
    )
)

model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = TImodel1(params; DDF = 2.0 / 1000.0, acc_factor = 1.4 / 1000.0)
)

glaciers = initialize_glaciers(rgi_ids, params)
prediction = Prediction(model, glaciers, params)
run!(prediction)

glacier = glaciers[1]
results = prediction.results[1]

folder = "results_plots"
mkpath(folder)

# ## Heatmaps

# The `plot_glacier` function with the `"heatmaps"` mode renders spatial maps of
# one or more glacier variables at a given time step.
# Common variables are `H` (ice thickness), `V` (velocity magnitude),
# `S` (surface elevation) and `B` (bed elevation).

fig_heatmaps = plot_glacier(results, "heatmaps", [:H, :V, :S, :B]; timeIdx = 1)
CairoMakie.save(joinpath(folder, "heatmaps.png"), fig_heatmaps)

# ```@raw html
# <img src="./results_plots/heatmaps.png" width="700"/>
# ```

# ## Velocity quivers

# To visualize glacier flow direction, use the `"quivers"` mode.
# Passing both `:V` and `:V_ref` will display the simulated and
# observed velocity fields side by side.

fig_quivers = plot_glacier(results, "quivers", [:V, :V_ref]; timeIdx = 1)
CairoMakie.save(joinpath(folder, "quivers.png"), fig_quivers)

# ```@raw html
# <img src="./results_plots/quivers.png" width="700"/>
# ```

# ## Evolution statistics

# The `"evolution statistics"` mode plots temporal statistics of a variable
# across the simulation. Any combination of `"average"`, `"median"`, `"min"` and
# `"max"` can be requested.

fig_stats = plot_glacier(
    results,
    "evolution statistics",
    [:H];
    tspan = results.tspan,
    metrics = ["average", "median", "min", "max"]
)
CairoMakie.save(joinpath(folder, "evolution_statistics.png"), fig_stats)

# ```@raw html
# <img src="./results_plots/evolution_statistics.png" width="700"/>
# ```

# ## Integrated volume

# The `"integrated volume"` mode shows how the total ice volume integrated over
# the glacier domain evolves throughout the simulation.

fig_volume = plot_glacier(results, "integrated volume", [:H]; tspan = results.tspan)
CairoMakie.save(joinpath(folder, "integrated_volume.png"), fig_volume)

# ```@raw html
# <img src="./results_plots/integrated_volume.png" width="700"/>
# ```

# ## Gridded data

# The `plot_gridded_data` function provides a lower-level interface for displaying
# any 2-D field on the glacier grid. It supports contour lines, log-scale plotting
# and custom color ranges.

fig_gridded = plot_gridded_data(
    results.S, results; title = "Surface elevation", colorbar_label = "m a.s.l."
)
CairoMakie.save(joinpath(folder, "gridded_surface.png"), fig_gridded)

# ```@raw html
# <img src="./results_plots/gridded_surface.png" width="700"/>
# ```

# ## Cumulative mass balance

# The `plot_cumulative_mb` function accumulates the gridded mass balance over time
# and renders a spatial map of the cumulative signal. Setting `annual_MB = true`
# produces the annually averaged equivalent.

fig_mb = plot_cumulative_mb(results; title = "Cumulative mass balance")
if fig_mb !== nothing
    CairoMakie.save(joinpath(folder, "cumulative_mb.png"), fig_mb)
end

# ```@raw html
# <img src="./results_plots/cumulative_mb.png" width="700"/>
# ```

# ## Digital elevation model

# A quick DEM overview of the glacier can be obtained from either a `Results`
# object or directly from a `Glacier2D` object.

fig_dem = plot_glacier_dem(results)
CairoMakie.save(joinpath(folder, "glacier_dem.png"), fig_dem)

# ```@raw html
# <img src="./results_plots/glacier_dem.png" width="700"/>
# ```

# ## Video

# The `plot_glacier_vid` function generates an `.mp4` animation of the ice
# thickness evolution throughout the simulation.

step_video = results.t[2] - results.t[1]
plot_glacier_vid(
    "thickness",
    results,
    glacier,
    results.tspan,
    step_video,
    joinpath(folder, "glacier_thickness.mp4");
    framerate = 12,
    baseTitle = "Ice thickness"
)

# ## Saving figures

# All figures can be saved with `save_figure`, which wraps `CairoMakie.save`
# and returns the output file path:

saved_path = save_figure(fig_dem, joinpath(folder, "glacier_dem_saved.png"))
