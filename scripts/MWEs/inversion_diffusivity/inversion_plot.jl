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

# using Pkg; Pkg.activate(".")
# include("inversion_setup.jl")

using Plots; pythonplot()
using LaTeXStrings
using JLD2

# res_load = load(joinpath(ODINN.root_dir, "scripts/MWEs/inversion_diffusivity/data", "simulation_result_test.jld2"), "res")
res_load = load(joinpath(ODINN.root_dir, "scripts/MWEs/inversion_diffusivity", "_inversion_result.jld2"), "res")


θ = res_load.θ
θ_hist = res_load.θ_hist
params = res_load.params

### Figures

# losses = functional_inversion.stats.losses

H_smooth = collect(1.0:1.0:maximum(only(glaciers).H₀))
∇S_smooth = collect(0.01:0.01:0.7)

function diffusivity_generate(θ, Hs, ∇Ss)

    D_pred = zeros(length(Hs), length(∇Ss))
    D_true = zeros(length(Hs), length(∇Ss))

    for i in 1:length(Hs), j in 1:length(∇Ss)
        # A_pred = ODINN.predict_A_target_D(architecture, T, [100.0, 200.0])
        h = Hs[i]
        ∇s = ∇Ss[j]
        _D = ODINN.Diffusivity_scalar(
            functional_inversion.model.machine_learning.target;
            h = h, ∇s = ∇s, θ = θ,
            iceflow_model = only(functional_inversion.model.iceflow),
            ml_model = functional_inversion.model.machine_learning,
            glacier = only(glaciers),
            params = functional_inversion.parameters)
        D_pred[i, j] = _D
        # Compute true diffusivity used for simulation
        A = halfar_params.A
        ρ = halfar_params.ρ
        g = halfar_params.g
        n = halfar_params.n
        D_true[i, j] = 2 * A * (ρ * g)^n * h^(n + 2) * ∇s^(n - 1) / (n + 2)
    end

    return D_true, D_pred

end

D_true, D_pred = diffusivity_generate(θ, H_smooth, ∇S_smooth)

idx_∇S = 10

plot = Plots.scatter(H_smooth, D_pred[:, idx_∇S], label="Neural network prediction", c=:lightsteelblue2);
Plots.plot!(H_smooth, D_true[:, idx_∇S], label="Ground True Value",
                    xlabel="Ice thickness H [m]",
                    # yscale = :log10,
                    ylabel="Predicted Diffusivity", lw = 3, c=:dodgerblue4,
                    legend=:topleft);
Plots.savefig(plot, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_test_H.pdf")

idx_H = 20

plot = Plots.scatter(∇S_smooth, D_pred[idx_H, :], label="Neural network prediction", c=:lightsteelblue2);
Plots.plot!(∇S_smooth, D_true[idx_H, :], label="Ground True Value",
                    xlabel="Ice surface slope [ratio]",
                    # yscale = :log10,
                    ylabel="Predicted Diffusivity", lw = 3, c=:dodgerblue4,
                    legend=:topleft);
Plots.savefig(plot, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_test_S.pdf")

### Make plot with curve lines in log scale

H = only(glaciers).H₀
S = only(glaciers).S
Δx, Δy = only(glaciers).Δx, only(glaciers).Δy
dSdx_edges = Huginn.diff_x(S) ./ Δx
dSdy_edges = Huginn.diff_y(S) ./ Δy
dSdx = zero(S)
dSdy = zero(S)
Huginn.inn(dSdx) .= Huginn.avg_x(dSdx_edges)[:, 2:end-1]
Huginn.inn(dSdy) .= Huginn.avg_y(dSdy_edges)[2:end-1, :]
∇S = (dSdx.^2 + dSdy.^2).^0.5

H_flat = vcat(H...)
∇S_flat = vcat(∇S...)
∇S_flat = ∇S_flat[H_flat .> 0.0]
H_flat = H_flat[H_flat .> 0.0]

# min_level = min(minimum(D_pred), minimum(D_true))
# max_level = max(maximum(D_pred), maximum(D_true))
min_level, max_level = minimum(D_true), maximum(D_true)
levels = floor(log10(min_level)):1.0:ceil(log10(max_level))

D_pred[D_pred .< 0.1 .* minimum(D_true)] .= NaN

plot_cont = Plots.contourf(H_smooth, ∇S_smooth, log10.(D_pred'), color=:plasma, alpha = 0.5,
    levels = levels, lw = 1, grid = false,
    clabels=true, cbar=true)
Plots.contour!(H_smooth, ∇S_smooth, log10.(D_true'), color=[:black],
    levels = levels, lw = 0.4,
    clabels=true, cbar=true)
title!(L"Plot of $\log_{10}(D)$")
xlabel!(L"Ice thickness $H$ [m]")
ylabel!(L"Surface slope $\| \nabla S \|$")
# histogram2d!(vcat(H...), vcat(∇S...), alpha = 0.4, 
#     bins=(100, 100), show_empty_bins=false,
#     color = cgrad(:grays, rev = true), cbar=false,
#     normalize=:pdf);
Plots.scatter!(H_flat, ∇S_flat, ms=0.2, color=:black, label=false)

Plots.savefig(plot_cont, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_contour.pdf")

for (i, _θ) in enumerate(θ_hist)

    D_true, D_pred = diffusivity_generate(_θ, H_smooth, ∇S_smooth)
    D_pred[D_pred .< 0.1 .* minimum(D_true)] .= NaN

    plot_cont = Plots.contourf(H_smooth, ∇S_smooth, log10.(D_pred'), color=:plasma, alpha = 0.5,
        levels = levels, lw = 1, grid = false,
        clabels=true, cbar=true)
    Plots.contour!(H_smooth, ∇S_smooth, log10.(D_true'), color=[:black],
        levels = levels, lw = 0.4,
        clabels=true, cbar=true)
    title!(L"Plot of $\log_{10}(D)$")
    xlabel!(L"Ice thickness $H$ [m]")
    ylabel!(L"Surface slope $\| \nabla S \|$")
    # histogram2d!(vcat(H...), vcat(∇S...), alpha = 0.4, 
    #     bins=(100, 100), show_empty_bins=false,
    #     color = cgrad(:grays, rev = true), cbar=false,
    #     normalize=:pdf);
    Plots.scatter!(H_flat, ∇S_flat, ms=0.2, color=:black, label=false)
    Plots.savefig(plot_cont, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_contour_epoch_$(i).pdf")
end