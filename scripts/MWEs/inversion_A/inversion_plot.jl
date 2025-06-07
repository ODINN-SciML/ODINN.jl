using Pkg; Pkg.activate(".")
include("inversion_setup.jl")

using Plots; pythonplot()
using JLD2

res_load = load(joinpath(ODINN.root_dir, "scripts/MWEs/inversion_A/data", "simulation_result.jld2"), "res")

losses = res_load.losses
θ = res_load.θ

Temps_smooth = collect(-23.0:0.1:0.0)
As_fake = fakeA.(Temps_smooth)

Temps = Float64[]
As_pred = Float64[]

for (i, glacier) in enumerate(glaciers)
    T = ODINN.mean(glacier.climate.longterm_temps)
    A = ODINN.apply_parametrization(
        functional_inversion.model.machine_learning.target;
        H = nothing, ∇S = nothing, θ = θ,
        iceflow_model = functional_inversion.model.iceflow[i],
        ml_model = functional_inversion.model.machine_learning,
        glacier = glacier,
        params = functional_inversion.parameters)
    push!(Temps, T)
    push!(As_pred, A)
end

Plots.plot(Temps_smooth, As_fake, label="True A", c=:lightsteelblue2)
plot_epoch = Plots.scatter!(Temps, As_pred, label="Predicted A", 
                    xlabel="Long-term air temperature (°C)", yticks=[0.0, 1e-17, 1e-18, params.physical.maxA],
                    ylabel=:A, ylims=(0.0, params.physical.maxA), lw = 3, c=:dodgerblue4,
                    legend=:topleft)


# @test As_pred ≈ As_fake rtol=0.1
# @test log10.(As_pred) ≈ log10.(As_fake) rtol=0.1

Plots.savefig(plot_epoch, "scripts/MWEs/inversion_A/figures/MWE_inversion_A.pdf")
