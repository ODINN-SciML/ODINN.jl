using Pkg
odinn_folder = split(Base.source_dir(), "scripts")[1]
Pkg.activate(odinn_folder*"/scripts/")

include("inversion_setup.jl")

# using Plots## pythonplot()
# using JLD2
import ODINN: load, Plots

res_load = load(joinpath(ODINN.root_dir, "scripts/MWEs/inversion_A/data", "simulation_result.jld2"), "res")

losses = res_load.losses
θ = res_load.θ
H₀_pred = if haskey(θ, :IC)
    [Matrix(θ.IC)]
else
    nothing
end

Temps_smooth = collect(-23.0:0.1:0.0)
A_poly = Huginn.polyA_PatersonCuffey()
As_fake = A_poly.(Temps_smooth)

Temps = Float64[]
As_pred = Float64[]
H₀_ref = Matrix[]
# H₀_pred = Matrix[]

for (i, glacier) in enumerate(glaciers)
    T, A = ODINN.T_A_Alaw(functional_inversion, i, θ, tstops[end])
    push!(Temps, T)
    push!(As_pred, A)
    push!(H₀_ref, glacier.H₀)
end

# For now we do this manually


Plots.plot(Temps_smooth, As_fake, label="True A", c=:lightsteelblue2)
plot_epoch = Plots.scatter!(Temps, As_pred, label="Predicted A",
                    xlabel="Long-term air temperature (°C)", yticks=[0.0, 1e-17, 1e-18, params.physical.maxA],
                    ylabel=:A, ylims=(0.0, params.physical.maxA), lw = 3, c=:dodgerblue4,
                    legend=:topleft)


# @test As_pred ≈ As_fake rtol=0.1
# @test log10.(As_pred) ≈ log10.(As_fake) rtol=0.1

mkpath("scripts/MWEs/inversion_A/figures")
Plots.savefig(plot_epoch, "scripts/MWEs/inversion_A/figures/MWE_inversion_A.pdf")

# plot_initial_condition = Plots.Figure(resolution = (800, 800))
# Plots.Axis(plot_initial_condition[1, 1], title = "Initial Condition")
# Plots.heatmap!(H₀_ref[1], colormap = :Blues,colorrange = (0.0, 200.0));

# save("scripts/MWEs/inversion_A/figures/MWE_inversion_A_initial_condition.pdf", plot_initial_condition)

# Plots.savefig(plot_initial_condition, "scripts/MWEs/inversion_A/figures/MWE_inversion_A_initial_condition.pdf")

plot_initial_condition = Plots.heatmap(H₀_ref[1], colormap = :Blues, clim = (0.0, 200.0));
Plots.savefig(plot_initial_condition, "scripts/MWEs/inversion_A/figures/MWE_inversion_A_initial_condition_ref.pdf")

plot_initial_condition = Plots.heatmap(H₀_pred[1], colormap = :Blues, clim = (0.0, 200.0));
Plots.savefig(plot_initial_condition, "scripts/MWEs/inversion_A/figures/MWE_inversion_A_initial_condition_pred.pdf")

plot_initial_condition = Plots.heatmap(H₀_ref[1] .- H₀_pred[1], colormap = :bwr, clim = (-10.0, 10.0));
Plots.savefig(plot_initial_condition, "scripts/MWEs/inversion_A/figures/MWE_inversion_A_initial_condition_diff.pdf")
