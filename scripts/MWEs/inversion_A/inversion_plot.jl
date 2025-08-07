using Pkg
odinn_folder = split(Base.source_dir(), "scripts")[1]
Pkg.activate(odinn_folder*"/scripts/")

include("inversion_setup.jl")

using Plots## pythonplot()
using JLD2

res_load = load(joinpath(ODINN.root_dir, "scripts/MWEs/inversion_A/data", "simulation_result.jld2"), "res")

losses = res_load.losses
θ = res_load.θ

Temps_smooth = collect(-23.0:0.1:0.0)
A_poly = Huginn.polyA_PatersonCuffey()
As_fake = A_poly.(Temps_smooth)

Temps = Float64[]
As_pred = Float64[]

for (i, glacier) in enumerate(glaciers)
    T, A = ODINN.T_A_Alaw(functional_inversion, i, θ, tstops[end])
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

mkpath("scripts/MWEs/inversion_A/figures")
Plots.savefig(plot_epoch, "scripts/MWEs/inversion_A/figures/MWE_inversion_A.pdf")
