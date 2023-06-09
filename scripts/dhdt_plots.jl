
# using Plots; gr()
using CairoMakie
using JLD2
import ODINN: fillZeros


function make_plots()

    # plot_type = "only_H" # plot final H
    # plot_type = "MB_diff" # differences between runs with different MB models
    plot_type = "H_diff" # H - H₀
    tspan = (2010.0, 2015.0) # period in years for simulation

    root_dir = dirname(Base.current_project())

    # Load forward simulations with different surface MB
    grefs = load(joinpath(root_dir, "data/gdir_refs_$tspan.jld2"))["gdir_refs"]
    grefs_MBu1 = load(joinpath(root_dir, "data/gdir_refs_updatedMB1.jld2"))["gdir_refs"]

    n=4
    m=3
    hms_MBdiff, MBdiffs = [], []
    figMB = Figure(resolution = (900, 1100))
    axsMB = [Axis(figMB[i, j]) for i in 1:n, j in 1:m]
    hidedecorations!.(axsMB)
    tightlimits!.(axsMB)
    let label=""
    for (i, ax) in enumerate(axsMB)
        ax.aspect = DataAspect()
        name = grefs[i]["RGI_ID"]
        ax.title = name
        H = reverse(grefs[i]["H"]', dims=2)
        H₀ = reverse(grefs[i]["H₀"]', dims=2)
        H_MBu1 = reverse(grefs_MBu1[i]["H"]', dims=2)
        # H = reverse(grefs[i]["H"])
        # H_MBu1 = reverse(grefs_MBu1[i]["H"])
        if plot_type == "only_H"
            H_plot = H
            label = "Predicted H (m)"
        elseif plot_type == "H_diff"
            H_plot = H .- H₀
            label = "H - H₀ (m)"
        elseif plot_type == "MB_diff"
            H_plot = H .- H_MBu1
            label="Surface mass balance difference (m)"
        end
        push!(MBdiffs, H_plot)
        push!(hms_MBdiff, CairoMakie.heatmap!(ax, fillZeros(H_plot), colormap=:inferno))
    end

    minMBdiff = minimum(minimum.(MBdiffs))
    maxMBdiff = maximum(maximum.(MBdiffs)) 
    foreach(hms_MBdiff) do hm
        hm.colorrange = (minMBdiff, maxMBdiff)
    end
    Colorbar(figMB[2:3,m+1], limits=(minMBdiff/2,maxMBdiff/2), label=label, colormap=:inferno)
    #Label(figH[0, :], text = "Glacier dataset", textsize = 30)
    if plot_type == "only_H"
        Makie.save(joinpath(root_dir, "plots/MB/H_MB_$tspan.pdf"), figMB, pt_per_unit = 1)
    elseif plot_type == "H_diff"
        Makie.save(joinpath(root_dir, "plots/MB/H_diff_wMB_$tspan.pdf"), figMB, pt_per_unit = 1)
    elseif plot_type == "MB_diff"
        Makie.save(joinpath(root_dir, "plots/MB/diffs_noMB_$tspan.pdf"), figMB, pt_per_unit = 1)
    end

    end # let

    # hms = []
    # for (gref, gref_MBu1) in zip(grefs, grefs_MBu1)
    #     H = reverse(gref["H"], dims=1)
    #     H_MBu1 = reverse(gref_MBu1["H"], dims=1)
    #     # H = gref["H"]
    #     # H_MBu1 = gref_MBu1["H"]
    #     push!(hms, heatmap(H .- H_MBu1, 
    #                         clims=(0.0,5.0),
    #                         ylimits=(0, size(H)[1]),
    #                         xlimits=(0, size(H)[2]),
    #                         colorbar = false)
    #         )
    # end

    # h2 = scatter([0,0], [0,1], clims=(0.0,5.0),
    #                  xlims=(1,1.1), xshowaxis=false, yshowaxis=false, label="", colorbar_title="cbar", grid=false)


    # l = @layout [grid(6,5) a{0.01w}]

    # # Create the combined plot with the subplots and shared colormap
    # p_dhdt = plot(hms..., h2,
    #               size=(1800, 1200),
    #               layout=l,
    #               link=:all,
    #               aspect_ratio=:equal)

    # savefig(p_dhdt, joinpath(root_dir, "plots/MB/dhdt_MB_1"))

end

make_plots()