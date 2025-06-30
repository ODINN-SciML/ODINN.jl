"""
The goal of this inversion is to recover the power law dependency of the diffusivity in
the SIA equation as a function of H.

We are going go generate a glacier flow equation following the next SIA diffusivity:

D(H, ∇S) = 2A / (n + 2) * (ρg)^n H^{n+2} |∇S|^{n-1}

and we are going to target this diffusivity with the function

D(H, ∇S, θ) = 2 / (n + 2) * H^2 |∇S|^{n-1} * NN(Temp, ρgH)

The neural network should learn the function NN(Temp, ρgH) ≈ (ρgH)^n

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
        epochs = [100, 20],
        optimizer = [ODINN.ADAM(0.01), ODINN.LBFGS()]
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
        target = :D_hybrid
        ),
    solver = Huginn.SolverParameters(
        step = δt,
        save_everystep = true,
        progress = true
        )
    )

# We retrieve some glaciers for the simulation
glaciers = initialize_glaciers(rgi_ids, params)

H_max = 1.3 * maximum(only(glaciers).H₀)

"""
The ground data was generated with n = 3.
In this inversion, we are interested if the network can learn a different Glen law that
the one it was prescribed.
We will prescribed then the ML model with n = 2:

We do this by providing a modifying the target object used in the NN

Inputs: Temp, H
"""
architecture = Lux.Chain(
    Dense(2, 3, x -> softplus.(x)),
    Dense(3, 3, x -> softplus.(x)),
    Dense(3, 1, sigmoid)
)

# The neural network shoudl return something between 0 and A * H^{max n power}
min_NN = 0.0
n_max = 2.3
max_NN = params.physical.maxA * H_max^n_max

min_temp, max_temp = - 25.0, 0.0
min_H, max_H = 0.0, H_max

# We define the prescale and postscale of quantities.
model = Huginn.Model(
    iceflow = SIA2Dmodel(params; A=CuffeyPaterson()),
    mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0),
)

# Time snapshots for transient inversion
tstops = collect(2010:δt:2015)

A_poly = Huginn.polyA_PatersonCuffey()

# We generate a fake law with A and no direct dependency on H
generate_ground_truth!(glaciers, params, model, tstops)

prediction = Huginn.Prediction(model, glaciers, params)

Temps = Float64[]
As_fake = Float64[]

for i in 1:length(glaciers)
    T, A = ODINN.T_A_Alaw(prediction, i, nothing, tstops[end])
    println("Value of A used to generate fake data: $(A)")
    push!(Temps, T)
    push!(As_fake, A)
end

# TODO: This function does shit on the model variable, for now we do a clean restart
nn_model = NeuralNetwork(
    params;
    architecture = architecture,
    target = SIA2D_D_hybrid_target(
        n_H = 1.0,
        max_NN = max_NN
    )
)
A_law = LawUhybrid(nn_model, params)
model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0),
    regressors = (; A=nn_model),
)

# We create an ODINN prediction
functional_inversion = FunctionalInversion(model, glaciers, params)

# We run the simulation with ADAM and then LBFGS
run!(functional_inversion)

### Figures

losses = functional_inversion.stats.losses

# Temps_smooth = collect(-23.0:1.0:0.0)
T₀ = Temps[1]
Temps_smooth = [T₀]
H_smooth = collect(0.0:1.0:max_H)

AtimesH_pred = zeros(length(Temps_smooth), length(H_smooth))

θ = functional_inversion.model.machine_learning.θ
for i in 1:length(Temps_smooth), j in 1:length(H_smooth)
    temp = Temps_smooth[i]
    H = H_smooth[j]

    A_pred = eval_law(functional_inversion.model.iceflow.A, functional_inversion, 1, (;T=temp, H̄=H), θ)
    AtimesH_pred[i, j] = only(unique(A_pred)) # The cache is a matrix and the result of the NN evaluation has been broadcasted to a matrix, we retrieve the only value
end

A₀ = A_poly.(T₀)
H_pred = AtimesH_pred[1, :] #./ A₀

plot = Plots.scatter(H_smooth, H_pred, label="Neural network prediction", c=:lightsteelblue2)
Plots.plot!(H_smooth, A₀ .* H_smooth.^2.0, label="Ground True Value",
                    xlabel="Ice thickness H [m]",
                    ylabel="Predicted output (= A(T) x H^2)", lw = 3, c=:dodgerblue4,
                    legend=:topleft)
Plots.savefig(plot, "MWE_inversion_diffusion_result_H_2.pdf")

# T₀ = Temps[1]


# Plots.scatter(Temps, As_fake, label="True A", c=:lightsteelblue2)
# plot_epoch = Plots.plot!(Temps_smooth, As_pred, label="Predicted A", 
#                     xlabel="Long-term air temperature (°C)", yticks=[0.0, 1e-17, 1e-18, params.physical.maxA],
#                     ylabel=:A, ylims=(0.0, params.physical.maxA), lw = 3, c=:dodgerblue4,
#                     legend=:topleft)


# @test As_pred ≈ As_fake rtol=0.1
# @test log10.(As_pred) ≈ log10.(As_fake) rtol=0.1

# Plots.savefig(plot_epoch, "MWE_custom_adjoint_results.pdf")
