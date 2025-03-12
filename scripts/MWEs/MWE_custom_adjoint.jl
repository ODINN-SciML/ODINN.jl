# using Pkg; Pkg.activate(".")

using Distributed

# @everywhere begin

using Revise
using Optimization
# using EnzymeCore
using Enzyme
using Test
using Statistics
using ODINN

rgi_paths = get_rgi_paths()

working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")

sensealg = ODINN.ZygoteAdjoint()
adtype = ODINN.NoAD()


# Define dummy grad
dummy_grad = function (du, u; simulation::Union{FunctionalInversion, Nothing}=nothing)
    du .= maximum(abs.(u)) .* rand(Float64, size(u))
end

params = Parameters(simulation = SimulationParameters(working_dir=working_dir,
                                                    use_MB=false,
                                                    velocities=true,
                                                    tspan=(2010.0, 2015.0),
                                                    multiprocessing=false,
                                                    workers=1,
                                                    light=false, # for now we do the simulation like this (a better name would be dense)
                                                    test_mode=true,
                                                    rgi_paths=rgi_paths),
                    hyper = Hyperparameters(batch_size=4,
                                            epochs=300,
                                            optimizer=ODINN.ADAM(0.03)),
                    UDE = UDEparameters(sensealg=sensealg, 
                                        optim_autoAD=adtype, 
                                        grad=dummy_grad, 
                                        optimization_method="AD+AD",
                                        target = "A"),
                    solver = Huginn.SolverParameters(save_everystep=true, progress=true)
                    )

## Retrieving simulation data for the following glaciers
# rgi_ids = ["RGI60-11.03638"] #, "RGI60-11.01450"]#, "RGI60-08.00213", "RGI60-04.04351"]
# rgi_ids = ["RGI60-11.01450"]#, "RGI60-11.03638"] #, "RGI60-04.04351"]
rgi_ids = ["RGI60-11.03638", 
            "RGI60-11.01450"]#, 
            # "RGI60-08.00213", 
            # "RGI60-04.04351", 
            # "RGI60-01.02170",
            # "RGI60-02.05098", 
            # "RGI60-01.01104",
            # "RGI60-01.09162", 
            # "RGI60-01.00570", 
            # "RGI60-04.07051",
            # "RGI60-07.00274", 
            # "RGI60-07.01323",  
            # "RGI60-01.17316"]

model = Model(iceflow = SIA2Dmodel(params),
                mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
                machine_learning = NeuralNetwork(params))

# We retrieve some glaciers for the simulation
glaciers = initialize_glaciers(rgi_ids, params)

# Picj a medium value of A
# A₀ =10^(0.5 * log10(params.physical.minA) + 0.5 * log10(params.physical.maxA))
# @show A₀

tstops = collect(2010:(1/12):2015)

function fakeA(T)
    Tmin = -30.0
    Tmax = 0.0
    return params.physical.minA + (T-Tmin) * (params.physical.maxA - params.physical.minA) / (Tmax-Tmin) 
end

Temps = Float64[]
As_fake = Float64[]

# We generate a fake forward model for the simulation
# @everywhere begin
# for glacier in glaciers
#     # A = A₀
#     T = mean(glacier.climate.longterm_temps)
#     A = fakeA(T)
#     push!(Temps, T)
#     push!(As_fake, A)
#     @show T, A
#     # A = fakeA(T)
#     generate_glacier_prediction!(glacier, params, model; A = A, tstops=tstops)
# end
# end

function generate_ground_truth(glacier, fakeA::Function)
    T = mean(glacier.climate.longterm_temps)
    A = fakeA(T)
    generate_glacier_prediction!(glacier, params, model; A = A, tstops=tstops)
end
# end

map(glacier -> generate_ground_truth(glacier, fakeA), glaciers)

# We create an ODINN prediction
functional_inversion = FunctionalInversion(model, glaciers, params)

# We run the simulation
run!(functional_inversion)

losses = functional_inversion.stats.losses

# batch_id = 1

Temps_smooth = -23:1:0
As_pred = Float64[]

for T in Temps_smooth
    A_pred = ODINN.apply_UDE_parametrization(functional_inversion.stats.θ, functional_inversion, T)
    push!(As_pred, A_pred)
end

Plots.scatter(Temps, As_fake, label="True A", c=:lightsteelblue2)
plot_epoch = Plots.plot!(Temps_smooth, pred_A, label="Predicted A", 
                    xlabel="Long-term air temperature (°C)",# yticks=yticks,
                    ylabel="A", ylims=(0.0, params.simulation.parameters.physical.maxA), lw = 3, c=:dodgerblue4,
                    legend=:topleft)

@test As_pred ≈ As_fake rtol=0.1
@test log10.(As_pred) ≈ log10.(As_fake) rtol=0.1