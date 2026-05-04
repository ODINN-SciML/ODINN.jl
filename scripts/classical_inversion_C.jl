# Classical gridded inversion for C (sliding coefficient)
# Copy of classical_inversion.jl, but for C inversion using LawC and GriddedInv(:C)

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using Revise
using ODINN
using SciMLSensitivity
using Optimization

const RGI_ID = "RGI60-11.03638"
const TSPAN = (2010.0, 2015.0)
const δt = 1 / 12  # monthly output

params = Parameters(
    simulation = SimulationParameters(
        use_MB = false,
        tspan = TSPAN,
        test_mode = false,
        multiprocessing = false,
        rgi_paths = get_rgi_paths(),
        gridScalingFactor = 4,
    ),
    hyper = Hyperparameters(
        batch_size = 1,
        epochs = [10, 30],
        optimizer = [
            ODINN.Adam(0.02),
            ODINN.LBFGS(
                linesearch = ODINN.LineSearches.BackTracking(iterations = 5),
            ),
        ],
    ),
    physical = PhysicalParameters(minC = 1e-4, maxC = 1e-2),
    UDE = UDEparameters(
        grad = SciMLSensitivityAdjoint(),
        sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP()),
        optim_autoAD = Optimization.AutoZygote(),
        empirical_loss_function = LossH(),
        target = :A,
    ),
    solver = Huginn.SolverParameters(step = δt),
)

glaciers = initialize_glaciers([RGI_ID], params)
midC = (params.physical.minC + params.physical.maxC) / 2

tstops = collect(TSPAN[1]:δt:TSPAN[2])
# Generate ground-truth ice thickness using SyntheticC (a physics-based, non-trainable law
# that maps CPDD + topographic roughness to C via a sigmoid). No θ is involved here.
C_law_gt = SyntheticC(params)
model_gt = Model(
    iceflow = SIA2Dmodel(params; C = C_law_gt),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
)
prediction = generate_ground_truth_prediction(glaciers, params, model_gt, tstops)
glaciers = prediction.glaciers

# Record the ground-truth C field produced by SyntheticC for later comparison
C_ground_truth = fill(midC, size(prediction.glaciers[1].H₀))
C_ground_truth[1:(end - 1), 1:(end - 1)] .= eval_law(
    prediction.model.iceflow.C,
    prediction,
    1,
    (; CPDD = get_input(iCPDD(), prediction, 1, tstops[1]),
       topo_roughness = get_input(iTopoRough(), prediction, 1, tstops[1])),
    nothing,
)
C_ground_truth[prediction.glaciers[1].H₀ .== 0] .= NaN

# Step 2: Classical inversion (gridded LawC, GriddedInv)
# Keep the standard ODINN target routing: with SciMLSensitivityAdjoint the optimized
# quantity is determined by the trainable law/regressor (`C` here), not by a custom
# manual target type.
trainable_model = GriddedInv(params, glaciers, :C)
C_law = LawC(params; scalar = false)
model = Model(
    iceflow = SIA2Dmodel(params; C = C_law),
    mass_balance = TImodel1(params; DDF = 6.0 / 1000.0, acc_factor = 1.2 / 1000.0),
    regressors = (; C = trainable_model),
)

inversion = Inversion(model, glaciers, params)
run!(inversion)

# Step 3: Retrieve and visualise results
θ = inversion.results.stats.θ

C = fill(midC, size(inversion.glaciers[1].H₀))
inn1(C) .= eval_law(inversion.model.iceflow.C, inversion, 1, (;), θ)
C[inversion.glaciers[1].H₀ .== 0] .= NaN

outdir = mkpath(joinpath(@__DIR__, "../plots"))

fig_inv = plot_gridded_data(C, inversion.results.simulation[1]; colormap = :YlGnBu, logPlot = true)
save_figure(fig_inv, joinpath(outdir, "classical_inversion_C_inverted.png"))

fig_gt = plot_gridded_data(
    C_ground_truth,
    inversion.results.simulation[1];
    colormap = :YlGnBu,
    logPlot = true,
)
save_figure(fig_gt, joinpath(outdir, "classical_inversion_C_ground_truth.png"))

println("Plots saved to: ", abspath(outdir))
