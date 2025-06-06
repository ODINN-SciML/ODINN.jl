"""
The goal of this inversion is to recover the power law dependency of the diffusivity in
the SIA equation as a function of H.

We are going go generate a glacier flow equation following the next SIA diffusivity:

D(H, ∇S) = 2A / (n + 2) * (ρg)^n H^{n+2} |∇S|^{n-1}

and we are going to target this diffusivity with the function

D(H, ∇S, θ) = H * NN(Temp, H, ∇S)

The neural network should learn the function NN(Temp, H, ∇S) ≈  2A / (n + 2) * (ρg)^n H^{n+1} |∇S|^{n-1}
which corresponds to the time integrated ice surface velocity.

with n the Glen exponent.
"""

using Pkg; Pkg.activate(".")

using Revise
using ODINN
using SciMLSensitivity
using Lux, ComponentArrays
using Statistics
using Plots

rgi_paths = get_rgi_paths()

# The value of this does not really matter, it is hardcoded in Sleipnir right now.
working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")
# Re-set global constant for working directory
# const global Sleipnir.prepro_dir = joinpath(homedir(),  ".OGGM/ODINN_tests")


## Retrieving simulation data for the following glaciers
# rgi_ids = collect(keys(rgi_paths))
# rgi_ids = ["RGI60-11.03646"]
rgi_ids = ["RGI60-08.00203"]

# TODO: Currently there are two different steps defined in params.simulationa and params.solver which need to coincide for manual discrete adjoint
δt = 1/12
# Time snapshots for transient inversion
tstops = collect(2010:δt:2015)

params = Parameters(
    simulation = SimulationParameters(
        working_dir = working_dir,
        use_MB = false,
        velocities = true,
        tspan = (2010.0, 2015.0),
        step = δt,
        multiprocessing = false,
        workers = 1,
        test_mode = false,
        rgi_paths = rgi_paths
        ),
    hyper = Hyperparameters(
        batch_size = length(rgi_ids), # We set batch size equals all datasize so we test gradient
        epochs = [10, 50],
        optimizer = [ODINN.ADAM(0.002), ODINN.LBFGS()]
        ),
    physical = PhysicalParameters(
        minA = 8e-21,
        maxA = 8e-17
        ),
    UDE = UDEparameters(
        sensealg = SciMLSensitivity.ZygoteAdjoint(), # QuadratureAdjoint(autojacvec=ODINN.EnzymeVJP()),
        optim_autoAD = ODINN.NoAD(),
        grad = ContinuousAdjoint(),
        optimization_method = "AD+AD",
        target = :D_pure
        ),
    solver = Huginn.SolverParameters(
        step = δt,
        save_everystep = true,
        progress = true
        )
    )

# We retrieve some glaciers for the simulation
glaciers = initialize_glaciers(rgi_ids, params)

"""
We can define the architecture of the model directly, passing the prescale and postcale
directly to Lux using a WrappedFunction layer.
"""
architecture = Lux.Chain(
    WrappedFunction(
        x -> [
            ODINN.normalize(x[1]; lims = (0.0, 200.0)),
            ODINN.normalize(x[2]; lims = (0.0, 1.0))
            ]
        ),
    Dense(2, 5, x -> softplus.(x)),
    Dense(5, 8, x -> softplus.(x)),
    Dense(8, 3, x -> softplus.(x)),
    Dense(3, 1, sigmoid),
    WrappedFunction(y -> 10.0 .* exp.((y .- 1.0) ./ y))
)

# We define the prescale and postscale of quantities.
model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0),
    machine_learning = NeuralNetwork(
        params;
        architecture = architecture,
        target = SIA2D_D_target(
            interpolation = :None,
            n_interp_half = 20,
            prescale_provided = true,
            postscale_provided = true
        )
    )
)

A_poly = ODINN.A_law_PatersonCuffey()
fakeA(T) = A_poly(T)

# We generate a fake law with A and no direct dependency on H
ODINN.generate_ground_truth(glaciers, :PatersonCuffey, params, model, tstops)

Temps = Float64[]
As_fake = Float64[]

for glacier in glaciers
    T = mean(glacier.climate.longterm_temps)
    A = fakeA(T)
    println("Value of A used to generate fake data: $(A)")
    push!(Temps, T)
    push!(As_fake, A)
end

# TODO: This function does shit on the model variable, for now we do a clean restart

model.iceflow = SIA2Dmodel(params)

# We create an ODINN prediction
functional_inversion = FunctionalInversion(model, glaciers, params)

# We run the simulation with ADAM and then LBFGS
run!(functional_inversion)

### Figures

losses = functional_inversion.stats.losses

# Temps_smooth = collect(-23.0:1.0:0.0)
T₀ = mean(glaciers[1].climate.longterm_temps)
Temps_smooth = [T₀]
H_smooth = collect(0.0:1.0:max_H)

AtimesH_pred = zeros(length(Temps_smooth), length(H_smooth))

for i in 1:length(Temps_smooth), j in 1:length(H_smooth)
    # A_pred = ODINN.predict_A_target_D(architecture, T, [100.0, 200.0])
    temp = Temps_smooth[i]
    H = H_smooth[j]
    A_pred = ODINN.predict_A(
        functional_inversion.model.machine_learning.target,
        functional_inversion.stats.θ, temp, H;
        ml_model = functional_inversion.model.machine_learning,
        params = functional_inversion.parameters)
    AtimesH_pred[i, j] = A_pred
end

A₀ = fakeA(T₀)
H_pred = AtimesH_pred[1, :] #./ A₀

plot = Plots.scatter(H_smooth, H_pred, label="Neural network prediction", c=:lightsteelblue2)
Plots.plot!(H_smooth, A₀ .* H_smooth.^2.0, label="Ground True Value",
                    xlabel="Ice thickness H [m]",
                    ylabel="Predicted output (= A(T) x H^2)", lw = 3, c=:dodgerblue4,
                    legend=:topleft)
Plots.savefig(plot, "MWE_inversion_diffusion_result_H_2.pdf")

# T₀ = mean(glaciers[1].climate.longterm_temps)


# Plots.scatter(Temps, As_fake, label="True A", c=:lightsteelblue2)
# plot_epoch = Plots.plot!(Temps_smooth, As_pred, label="Predicted A", 
#                     xlabel="Long-term air temperature (°C)", yticks=[0.0, 1e-17, 1e-18, params.physical.maxA],
#                     ylabel=:A, ylims=(0.0, params.physical.maxA), lw = 3, c=:dodgerblue4,
#                     legend=:topleft)


# @test As_pred ≈ As_fake rtol=0.1
# @test log10.(As_pred) ≈ log10.(As_fake) rtol=0.1

# Plots.savefig(plot_epoch, "MWE_custom_adjoint_results.pdf")