
# using Plots; gr()
using CairoMakie
using JLD2
import ODINN: fillZeros


function make_plots(tspan, nglaciers)

    # plot_type = "only_H" # plot final H
    # plot_type = "MB_diff" # differences between runs with different MB models
    # plot_type = "H_diff" # H - H₀
    plot_type = "S"
    # tspan = (1990.0, 2015.0) # period in years for simulation
    # nglaciers = 5

    root_dir = dirname(Base.current_project())

    # Load forward simulations with different surface MB
    grefs = load(joinpath(root_dir, "data/results/predictions/prediction_$(nglaciers)glaciers_$tspan.jld2"))["results"]
    # grefs_MBu1 = load(joinpath(root_dir, "data/gdir_refs_$(tspan)_MB1.jld2"))["gdir_refs"]
    # grefs_MBu2 = load(joinpath(root_dir, "data/gdir_refs_$(tspan)_MB2.jld2"))["gdir_refs"]
    # grefs_MBu3 = load(joinpath(root_dir, "data/gdir_refs_$(tspan)_MB3.jld2"))["gdir_refs"]

    n=2
    m=2
    hms_MBdiff, MBdiffs = [], []
    figMB = Figure(resolution = (1000, 500))
    axsMB = [Axis(figMB[i, j]) for i in 1:n, j in 1:m]
    hidedecorations!.(axsMB)
    tightlimits!.(axsMB)
    let label=""
    for (i, ax) in enumerate(axsMB)
        ax.aspect = DataAspect()
        name = grefs[i].rgi_id
        ax.title = name
        H = reverse(grefs[i].H[end]', dims=2)
        H₀ = reverse(grefs[i].H[begin]', dims=2)
        S = reverse(grefs[i].S', dims=2)
        # H_MBu1 = reverse(grefs_MBu1[i]["H"]', dims=2)
        # H_MBu2 = reverse(grefs_MBu3[i]["H"]', dims=2)
        if plot_type == "only_H"
            H_plot = H
            label = "Predicted H (m)"
        elseif plot_type == "H_diff"
            H_plot = H .- H₀
            label = "H - H₀ (m)"
        elseif plot_type == "MB_diff"
            H_plot = H_MBu1 .- H_MBu2
            label="Surface mass balance difference (m)"
        elseif plot_type == "S"
        H_plot = S
        label="Surface elevation (m)"
        end
        push!(MBdiffs, H_plot)
        push!(hms_MBdiff, CairoMakie.heatmap!(ax, fillZeros(H_plot), colormap=:inferno))
    end

    minMBdiff = minimum(minimum.(MBdiffs))
    maxMBdiff = maximum(maximum.(MBdiffs)) 
    foreach(hms_MBdiff) do hm
        hm.colorrange = (minMBdiff, maxMBdiff)
    end
    Colorbar(figMB[1:2,m+1], limits=(minMBdiff/2,maxMBdiff/2), label=label, colormap=:inferno)
    supertitle = Label(figMB[0, :], "$(Int(tspan[1]))-$(Int(tspan[2]))", fontsize = 20)
    #Label(figH[0, :], text = "Glacier dataset", textsize = 30)
    if plot_type == "only_H"
        Makie.save(joinpath(root_dir, "plots/MB/H_MB_$tspan.pdf"), figMB, pt_per_unit = 1)
    elseif plot_type == "H_diff"
        Makie.save(joinpath(root_dir, "plots/MB/H_diff_wMB_$tspan.pdf"), figMB, pt_per_unit = 1)
    elseif plot_type == "MB_diff"
        Makie.save(joinpath(root_dir, "plots/MB/diffs_noMB_$tspan.pdf"), figMB, pt_per_unit = 1)
    elseif plot_type == "S"
        Makie.save(joinpath(root_dir, "plots/MB/S_$tspan.pdf"), figMB, pt_per_unit = 1)
    end

    end # let

end
