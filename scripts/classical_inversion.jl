using Pkg

Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Revise
using ODINN
using SciMLSensitivity
using Optimization

# ── Glacier and time settings ────────────────────────────────────────────────── #
const RGI_ID = "RGI60-11.03638"
const TSPAN = (2010.0, 2015.0)
const δt = 1 / 12  # monthly output

# ── Parameters ───────────────────────────────────────────────────────────────── #
params = Parameters(
    simulation = SimulationParameters(
        use_MB = false,
        tspan = TSPAN,
        test_mode = false,
        multiprocessing = false,
        rgi_paths = get_rgi_paths(),
        gridScalingFactor = 4
    ),
    hyper = Hyperparameters(
        batch_size = 1,
        epochs = [10, 30],
        optimizer = [
            ODINN.Adam(0.02),
            ODINN.LBFGS(
                linesearch = ODINN.LineSearches.BackTracking(iterations = 5),
            )
        ]
    ),
    physical = PhysicalParameters(minA = 8e-21, maxA = 8e-17),
    UDE = UDEparameters(
        grad = SciMLSensitivityAdjoint(),
        sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP()),
        optim_autoAD = Optimization.AutoZygote(),
        empirical_loss_function = LossH()
    ),
    solver = Huginn.SolverParameters(step = δt)
)

glaciers = initialize_glaciers([RGI_ID], params)

# ── Step 1: Generate synthetic ground truth via forward simulation ────────────── #
A_law = CuffeyPaterson(scalar = false)
model = Model(
    iceflow = SIA2Dmodel(params; A = A_law),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0)
)

tstops = collect(TSPAN[1]:δt:TSPAN[2])
prediction = generate_ground_truth_prediction(glaciers, params, model, tstops)
glaciers = prediction.glaciers

# Ground-truth A for comparison
A_ground_truth = zeros(size(prediction.glaciers[1].H₀))
A_ground_truth[1:(end - 1), 1:(end - 1)] .= eval_law(
    prediction.model.iceflow.A,
    prediction,
    1,
    (; T = get_input(iAvgGriddedTemp(), prediction, 1, tstops[1])),
    nothing
)
A_ground_truth[prediction.glaciers[1].H₀ .== 0] .= NaN

# ── Step 2: Classical inversion ──────────────────────────────────────────────── #
trainable_model = GriddedInv(params, glaciers, :A)
A_law = LawA(params; scalar = false)
model = Model(
    iceflow = SIA2Dmodel(params; A = A_law),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
    regressors = (; A = trainable_model)
)

inversion = Inversion(model, glaciers, params)
run!(inversion)

# ── Step 3: Retrieve and visualise results ───────────────────────────────────── #
θ = inversion.results.stats.θ

A = zeros(size(inversion.glaciers[1].H₀))
inn1(A) .= eval_law(inversion.model.iceflow.A, inversion, 1, (;), θ)
A[inversion.glaciers[1].H₀ .== 0] .= NaN

outdir = mkpath(joinpath(@__DIR__, "../plots"))

fig_inv = plot_gridded_data(A, inversion.results.simulation[1]; colormap = :YlGnBu, logPlot = true)
save_figure(fig_inv, joinpath(outdir, "classical_inversion_A_inverted.png"))

fig_gt = plot_gridded_data(
    A_ground_truth,
    inversion.results.simulation[1];
    colormap = :YlGnBu,
    logPlot = true
)
save_figure(fig_gt, joinpath(outdir, "classical_inversion_A_ground_truth.png"))

println("Plots saved to: ", abspath(outdir))
