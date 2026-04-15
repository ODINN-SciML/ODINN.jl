using Pkg

Pkg.activate(normpath(joinpath(@__DIR__, "..")))

# Point unreleased ecosystem packages to local dev checkouts.
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Muninn")
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Sleipnir")
Pkg.develop(path = "/Users/Bolib001/.julia/dev/Huginn")
Pkg.develop(path = "/Users/Bolib001/Desktop/Jordi/Julia/MassBalanceMachine.jl")

using Revise
using ODINN
using MassBalanceMachine

# Glacier: Hardangerjokulen, Norway (RGI60-08.00203)
const RGI_ID = "RGI60-08.00203"
const TSPAN = (2010.0, 2012.0)
const MBM_DIR = "/Users/Bolib001/Desktop/Jordi/Julia/MassBalanceMachine.jl/data/geo_20260205_180505_wgeo=0_scaling"

# ── Build simulation parameters ─────────────────────────────────────────────── #
params = Parameters(
    simulation = SimulationParameters(
        working_dir = Sleipnir.prepro_dir,
        tspan = TSPAN,
        multiprocessing = false,
        workers = 1,
        climate_data_source = :ERA5,
        rgi_paths = get_rgi_paths(),
        use_MB = true,
        use_iceflow = true,
        test_mode = false
    ),
    solver = SolverParameters(
        step = 1 / 12,
        progress = true,
        save_everystep = false
    )
)

# ── Assemble model ───────────────────────────────────────────────────────────── #
mb_model = CustomMLP(joinpath(MBM_DIR, "params.json"), joinpath(MBM_DIR, "best_model.json"))

model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = mb_model
)

# ── Run forward simulation ───────────────────────────────────────────────────── #
glaciers = initialize_glaciers([RGI_ID], params)
prediction = Prediction(model, glaciers, params)
run!(prediction)

# ── Visualise results ────────────────────────────────────────────────────────── #
plot_glacier(prediction.results[1], "evolution difference", [:H]; metrics = ["difference"])

fig = plot_cumulative_mb(prediction.results[1]; colormap = :balance, plotContour = true)
if !isnothing(fig)
    save_figure(fig, joinpath(ODINN.root_plots, "cumulative_mb_hardangerjokulen.png"))
end
