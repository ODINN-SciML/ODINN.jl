using Pkg; Pkg.activate(".")

using Distributed

# @everywhere begin

using Revise
using Optimization
# using EnzymeCore
using Enzyme
using Test
using Statistics
using ODINN
using Polynomials
using Plots
using SciMLSensitivity

Enzyme.API.strictAliasing!(false)

rgi_paths = get_rgi_paths()

# The value of this does not really matter, it is hardcoded in Sleipnir right now.
working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")
# Re-set global constant for working directory
# const global Sleipnir.prepro_dir = joinpath(homedir(),  ".OGGM/ODINN_tests")


## Retrieving simulation data for the following glaciers
rgi_ids = ["RGI60-11.03638"]
# rgi_ids = ["RGI60-08.00213", "RGI60-02.05098"]
# rgi_ids = [#"RGI60-11.03638",
#             # "RGI60-11.01450",
#             "RGI60-08.00213",
#             "RGI60-04.04351",
#             "RGI60-01.02170",
#             "RGI60-02.05098",
#             # "RGI60-01.01104",
#             # "RGI60-01.09162",
#             # "RGI60-01.00570", # This one does not have millan_v data
#             # "RGI60-04.07051",
#             "RGI60-07.00274",
#             "RGI60-07.01323"]#,
            # "RGI60-01.17316"] # This one does not have millan_v data


# TODO: Currently there are two different steps defined in params.simulationa and params.solver which need to coincide for manual discrete adjoint
δt = 1/12

params = Parameters(simulation = SimulationParameters(working_dir=working_dir,
                                                    use_MB=false,
                                                    velocities=true,
                                                    tspan=(2010.0, 2015.0),
                                                    step=δt,
                                                    multiprocessing=false,
                                                    workers=1,
                                                    light=false, # for now we do the simulation like this (a better name would be dense)
                                                    test_mode=true,
                                                    rgi_paths=rgi_paths),
                    hyper = Hyperparameters(batch_size=length(rgi_ids), # We set batch size equals all datasize so we test gradient
                                            epochs=100,
                                            # optimizer=ODINN.ADAM(0.005)),
                                            optimizer=ODINN.Descent(0.01)),
                    UDE = UDEparameters(sensealg=SciMLSensitivity.ZygoteAdjoint(),
                                        optim_autoAD=ODINN.NoAD(),
                                        grad=DiscreteAdjoint(),
                                        optimization_method="AD+AD",
                                        target = "A"),
                    solver = Huginn.SolverParameters(step=δt,
                                                     save_everystep=true, 
                                                     progress=true)
                    )

model = Model(iceflow = SIA2Dmodel(params),
                mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
                machine_learning = NeuralNetwork(params))

# We retrieve some glaciers for the simulation
glaciers = initialize_glaciers(rgi_ids, params)

# Time stanpshots for transient inversion
tstops = collect(2010:δt:2015)

# A_poly = ODINN.A_law_PatersonCuffey()
# fakeA(T) = A_poly(T)

# Overwrite constant A fake function for testing
fakeA(T) = 2.21e-18

map(glacier -> ODINN.generate_ground_truth(glacier, fakeA, params, model, tstops), glaciers)
# TODO: This function does shit on the model variable, for now we do a clean restart
model.iceflow = SIA2Dmodel(params)

# We create an ODINN prediction
functional_inversion = FunctionalInversion(model, glaciers, params)

# We run the simulation with ADAM
run!(functional_inversion)
# # We do a second run with BFGS
# params.hyper.optimizer = ODINN.LBFGS() #ODINN.Optim.BFGS(; initial_stepnorm=0.01, linesearch=ODINN.LineSearches.BackTracking())
# params.hyper.epochs = 100
# # Pre-trained parameter
# θ_trained = functional_inversion.stats.θ
# model.machine_learning.θ = θ_trained
# # Reconstruct functional inversion based on pretrained parameter
# functional_inversion = FunctionalInversion(model, glaciers, params)
# run!(functional_inversion)

### Figures

losses = functional_inversion.stats.losses

Temps_smooth = collect(-23.0:1.0:0.0)
As_pred = Float64[]

for T in Temps_smooth
    A_pred = ODINN.apply_UDE_parametrization(functional_inversion.stats.θ, functional_inversion, T)
    push!(As_pred, A_pred)
end

Temps = Float64[]
As_fake = Float64[]

for glacier in glaciers
    T = mean(glacier.climate.longterm_temps)
    A = fakeA(T)
    push!(Temps, T)
    push!(As_fake, A)
end

Plots.scatter(Temps, As_fake, label="True A", c=:lightsteelblue2)
plot_epoch = Plots.plot!(Temps_smooth, As_pred, label="Predicted A", 
                    xlabel="Long-term air temperature (°C)", yticks=[0.0, 1e-17, 1e-18, params.physical.maxA],
                    ylabel="A", ylims=(0.0, params.physical.maxA), lw = 3, c=:dodgerblue4,
                    legend=:topleft)


# @test As_pred ≈ As_fake rtol=0.1
# @test log10.(As_pred) ≈ log10.(As_fake) rtol=0.1

# Plots.savefig(plot_epoch, "MWE_custom_adjoint_results.pdf")