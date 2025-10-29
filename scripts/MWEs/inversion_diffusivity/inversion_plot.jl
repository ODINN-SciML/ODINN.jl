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

using Pkg
odinn_folder = split(Base.source_dir(), "scripts")[1]
Pkg.activate(odinn_folder*"/scripts/")

include("inversion_setup.jl")

using Plots; pythonplot()
using LaTeXStrings
using JLD2
using ForwardDiff
using LinearAlgebra
using Random, Distributions

res_load = load(joinpath(ODINN.root_dir, "scripts/MWEs/inversion_diffusivity/data", "simulation_result_Halfar.jld2"), "res")
# res_load = load(joinpath(ODINN.root_dir, "scripts/MWEs/inversion_diffusivity", "_inversion_result_working.jld2"), "res")

# Load parameters of the trained neural network
θ = res_load.θ
θ_hist = res_load.θ_hist
params = res_load.params

### Figures

# losses = functional_inversion.results.stats.losses

# Smooth variable for plotting
H_smooth = collect(1.0:1.0:maximum(only(glaciers).H₀))
∇S_smooth = collect(0.01:0.01:0.4)
R_smooth = collect(0.0:10.0:(halfar_params.R₀))

"""
Function to generate predicion and true diffusivity used in diffusivity
"""
function diffusivity_generate(functional_inversion, θ, Hs, ∇Ss)

    D_pred = zeros(length(Hs), length(∇Ss))
    D_true = zeros(length(Hs), length(∇Ss))

    for i in 1:length(Hs), j in 1:length(∇Ss)
        h = Hs[i]
        ∇s = ∇Ss[j]
        inputs = (; H̄=h, ∇S=∇s)
        U = eval_law(functional_inversion.model.iceflow.U, functional_inversion, 1, inputs, θ)
        D_pred[i, j] = only(unique(U)) * h # The cache is a matrix and the result of the NN evaluation has been broadcasted to a matrix, we retrieve the only value
        # Compute true diffusivity used for simulation
        (; A, ρ, g, n) = halfar_params
        D_true[i, j] = 2 * A * (ρ * g)^n * h^(n + 2) * ∇s^(n - 1) / (n + 2)
    end

    return D_true, D_pred

end

# Generate matrix of prediction and reference
D_true, D_pred = diffusivity_generate(functional_inversion, θ, H_smooth, ∇S_smooth)

### Preprocessing for figures

H_flat = []
∇S_flat = []
### Flatten sample points
for i in 1:length(only(glaciers).thicknessData.H)
    _H = only(glaciers).thicknessData.H[i]
    # H = only(glaciers).H₀
    _B = only(glaciers).B
    _S = _B + _H
    _Δx, _Δy = only(glaciers).Δx, only(glaciers).Δy
    dSdx_edges = Huginn.diff_x(_S) ./ _Δx
    dSdy_edges = Huginn.diff_y(_S) ./ _Δy
    dSdx = zero(_S)
    dSdy = zero(_S)
    Huginn.inn(dSdx) .= Huginn.avg_x(dSdx_edges)[:, 2:end-1]
    Huginn.inn(dSdy) .= Huginn.avg_y(dSdy_edges)[2:end-1, :]
    _∇S = (dSdx.^2 + dSdy.^2).^0.5
    append!(H_flat, vec(_H))
    append!(∇S_flat, vec(_∇S))
end

# Remove zeros from ice thickness
∇S_flat = ∇S_flat[H_flat .> 0.0]
H_flat = H_flat[H_flat .> 0.0]
# Subsample for plotting
N_sample = 1000
zipped = collect(zip(H_flat, ∇S_flat))
sample_pairs = sample(zipped, N_sample; replace = false)
H_flat, ∇S_flat = map(x -> getindex.(sample_pairs, x), (1, 2))

# We can analyically compute this curve. Let's do it for some intermediate time
t_mean = sum(params.simulation.tspan) / 2
H_analytical = map(R -> halfar(R, 0.0, t_mean), R_smooth)
∇S_analytical = map(R -> abs(ForwardDiff.derivative(r -> halfar(r, 0.0, t_mean), R)), R_smooth)

### Figure: Contour plot with predicted and target diffusivity

min_level, max_level = minimum(D_true), maximum(D_true)
levels = floor(log10(min_level)):1.0:ceil(log10(max_level))

D_pred[D_pred .< 0.1 .* minimum(D_true)] .= NaN

plot_cont = Plots.contourf(
    H_smooth, ∇S_smooth, log10.(D_pred'), color=:plasma, alpha = 0.5,
    levels = levels, lw = 1, grid = false,
    clabels=true, cbar=true
    )
Plots.contour!(
    H_smooth, ∇S_smooth, log10.(D_true'), color=[:black],
    levels = levels, lw = 0.4,
    clabels=true, cbar=true
    )
title!(L"Plot of $\log_{10}(D)$")
xlabel!(L"Ice thickness $H$ [m]")
ylabel!(L"Surface slope $\| \nabla S \|$")
Plots.scatter!(H_flat, ∇S_flat, ms=0.2, color=:black, label=false)
Plots.plot!(H_analytical, ∇S_analytical, color=:black, linewidth=1.0)

mkpath("scripts/MWEs/inversion_diffusivity/figures")
Plots.savefig(plot_cont, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_contour.pdf")

### Figure: Same plot but now for the whole history during training

# for (i, _θ) in enumerate(θ_hist)

#     D_true, D_pred = diffusivity_generate(functional_inversion, _θ, H_smooth, ∇S_smooth)
#     D_pred[D_pred .< 0.1 .* minimum(D_true)] .= NaN

#     plot_cont = Plots.contourf(
#         H_smooth, ∇S_smooth, log10.(D_pred'),
#         color=:plasma, alpha = 0.5,
#         levels = levels, lw = 1, grid = false,
#         clabels=true, cbar=true
#         )
#     Plots.contour!(
#         H_smooth, ∇S_smooth, log10.(D_true'),
#         color=[:black],
#         levels = levels, lw = 0.4,
#         clabels=true, cbar=true
#         )
#     title!(L"Plot of $\log_{10}(D)$")
#     xlabel!(L"Ice thickness $H$ [m]")
#     ylabel!(L"Surface slope $\| \nabla S \|$")
#     Plots.scatter!(H_flat, ∇S_flat, ms=0.2, color=:black, label=false)
#     Plots.savefig(plot_cont, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_contour_epoch_$(i).pdf")
# end

### Figure: Along the trajectory
# The single Halfar solution has a very specific function for ∇S as a function of H.
# We can plot along this trajectory

_D_true, _D_pred = diffusivity_generate(functional_inversion, θ, H_analytical, ∇S_analytical)
D_true_analytical = diag(_D_true)
D_pred_analytical = diag(_D_pred)

plot_analytical = Plots.scatter(
    H_analytical, D_pred_analytical, label="Neural network prediction", c=:lightsteelblue2
    );
Plots.plot!(
    H_analytical, D_true_analytical, label="Ground True Value",
    xlabel="Ice thickness [m]",
    # yscale = :log10,
    ylabel="Predicted Diffusivity", lw = 3, c=:dodgerblue4,
    legend=:topleft
    );
Plots.plot!(
    twinx(), H_analytical, ∇S_analytical,
    label="Surface slope", ylabel = "Slope", c=:orange, lw=1
    );
Plots.savefig(plot_analytical, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_test_analytical.pdf")

### Figure: Value of D along specific values of H and ∇S

idx_∇S = 10

plot = Plots.scatter(
    H_smooth, D_pred[:, idx_∇S],
    label="Neural network prediction", c=:lightsteelblue2
    );
Plots.plot!(
    H_smooth, D_true[:, idx_∇S], label="Ground True Value",
    xlabel="Ice thickness H [m]",
    # yscale = :log10,
    ylabel="Predicted Diffusivity", lw = 3, c=:dodgerblue4,
    legend=:topleft
    );
Plots.savefig(plot, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_test_H.pdf")

idx_H = 20

plot = Plots.scatter(∇S_smooth, D_pred[idx_H, :], label="Neural network prediction", c=:lightsteelblue2);
Plots.plot!(
    ∇S_smooth, D_true[idx_H, :], label="Ground True Value",
    xlabel="Ice surface slope [ratio]",
    # yscale = :log10,
    ylabel="Predicted Diffusivity", lw = 3, c=:dodgerblue4,
    legend=:topleft
    );
Plots.savefig(plot, "scripts/MWEs/inversion_diffusivity/figures/MWE_inversion_diffusion_test_S.pdf")
