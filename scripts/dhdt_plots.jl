
# using Plots; gr()
using CairoMakie
using JLD2
import ODINN: fillZeros

plot_only_H = false
tspan = (2014.0, 2018.0) # period in years for simulation

root_dir = dirname(Base.current_project())

# Load forward simulations with different surface MB
grefs = load(joinpath(root_dir, "data/gdir_refs_$tspan.jld2"))["gdir_refs"]
# grefs_MBu1 = load(joinpath(root_dir, "data/gdir_refs_updatedMB1.jld2"))["gdir_refs"]

n=4
m=3
hms_MBdiff, MBdiffs = [], []
figMB = Figure(resolution = (900, 1100))
axsMB = [Axis(figMB[i, j]) for i in 1:n, j in 1:m]
hidedecorations!.(axsMB)
tightlimits!.(axsMB)

for (i, ax) in enumerate(axsMB)
    ax.aspect = DataAspect()
    name = grefs[i]["RGI_ID"]
    ax.title = name
    H = reverse(grefs[i]["H"]', dims=2)
    H₀ = reverse(grefs[i]["H_initial"]', dims=2)

    # We cap very small numbers for visualization.
    # It may be useful to do this directly at the END of the integrator.
    ϵ = 0.0001
    H[H.<ϵ] .= 0.0

    println("Mass gain: ", (sum(H) - sum(H₀)) / sum(H₀) * 100 )
    # H_MBu1 = reverse(grefs_MBu1[i]["H"]', dims=2)
    # H = reverse(grefs[i]["H"])
    # H_MBu1 = reverse(grefs_MBu1[i]["H"])
    if plot_only_H
        H_plot = H
    else
        H_plot = H .- H₀
        # H_plot = H .- H_MBu1
    end
    push!(MBdiffs, H_plot)
    if plot_only_H
        push!(hms_MBdiff, CairoMakie.heatmap!(ax, fillZeros(H_plot), colormap=:inferno))
    else
        push!(hms_MBdiff, CairoMakie.heatmap!(ax, H_plot, colormap=Reverse(:balance)))
    end
end


foreach(hms_MBdiff) do hm
    if plot_only_H
        minMBdiff = minimum(minimum.(MBdiffs))
        maxMBdiff = maximum(maximum.(MBdiffs)) 
        hm.colorrange = (minMBdiff, maxMBdiff)
        # be careful with the limits used in the colorbar of plots and colorbar
        Colorbar(figMB[2:3,m+1], limits=(minMBdiff/2,maxMBdiff/2), label="Surface mass balance difference (m)", colormap=:inferno)
    else
        hm.colorrange = (-40, 40)
        Colorbar(figMB[2:3,m+1], label="Surface mass balance difference (m)", colormap=Reverse(:balance), colorrange=(-40,40))
    end
end


if plot_only_H
    Makie.save(joinpath(root_dir, "plots/MB/diffs_noMB_$tspan.pdf"), figMB, pt_per_unit = 1)
else
    Makie.save(joinpath(root_dir, "plots/MB/diffs_noMB_$tspan.pdf"), figMB, pt_per_unit = 1)
end



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