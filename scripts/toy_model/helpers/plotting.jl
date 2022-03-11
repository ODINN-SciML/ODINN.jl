function plot_glacier_dataset(gdirs_climate, PDE_refs)
    # gdirs_climate = (dates, gdirs, longterm_temps, annual_temps)
    gdirs = gdirs_climate[2]
    theme = Theme(fontsize=18, font="TeX Gyre Heros")
    set_theme!(theme)
    # Ice surface velocities
    figV = Makie.Figure(resolution=(1200, 1800))
    n = 5
    m = floor(Int, length(gdirs_climate[2])/n)
    axsV = [Axis(figV[i, j]) for i in 1:n, j in 1:m]
    hidedecorations!.(axsV)
    tightlimits!.(axsV)
    Vs, hm_Vs = [],[]
    for (i, ax) in enumerate(axsV)
        ax.aspect = DataAspect()
        glacier_gd = xr.open_dataset(gdirs[i].get_filepath("gridded_data"))
        nx = glacier_gd.y.size # glacier extent
        ny = glacier_gd.x.size # really weird, but this is inversed 
        Δx = abs(gdirs[i].grid.dx)
        Δy = abs(gdirs[i].grid.dy)
        rgi_id = gdirs[i].rgi_id
        name = gdirs[i].name
        ax.title = "$name - $rgi_id"
        Vx = PDE_refs["V̄x_refs"][i]
        Vy = PDE_refs["V̄y_refs"][i]
        V = (abs.(Vx) .+ abs.(Vy))./2
        push!(Vs,V)
        push!(hm_Vs, Makie.heatmap!(ax, (1:nx).*Δx, (1:ny).*Δy, fillZeros(V), colormap=:speed))
    end
    minV = minimum(minimum.(Vs))
    maxV = maximum(maximum.(Vs)) 
    Colorbar(figV[2:3,m+1], limits=(minV,maxV), colormap=:speed, label="Ice surface velocity")
    Label(figV[0, :], text = "Glacier dataset", textsize = 30)
    display(figV)
    Makie.save(joinpath(root_plots, "glaciers_V.pdf"), figV, pt_per_unit = 1)

    # Ice thickness
    figH = Makie.Figure(resolution=(1200, 1800))
    axsH = [Axis(figH[i, j]) for i in 1:n, j in 1:m]
    hidedecorations!.(axsH)
    tightlimits!.(axsH)
    Hs, hm_Hs = [],[]
    for (i, ax) in enumerate(axsH)
        ax.aspect = DataAspect()
        glacier_gd = xr.open_dataset(gdirs[i].get_filepath("gridded_data"))
        nx = glacier_gd.y.size # glacier extent
        ny = glacier_gd.x.size # really weird, but this is inversed 
        Δx = abs(gdirs[i].grid.dx)
        Δy = abs(gdirs[i].grid.dy)
        rgi_id = gdirs[i].rgi_id
        name = gdirs[i].name
        ax.title = "$name - $rgi_id"
        H = glacier_gd.consensus_ice_thickness.data # initial ice thickness conditions for forward model
        push!(Hs, fillNaN(H))
        push!(hm_Hs, Makie.heatmap!(ax, H, colormap=:ice))
    end
    minH = minimum(minimum.(Hs))
    maxH = maximum(maximum.(Hs)) 
    Colorbar(figH[2:3,m+1], limits=(minH,maxH), colormap=:ice, label="Ice thickness (m)")
    Label(figH[0, :], text = "Glacier dataset", textsize = 30)
    display(figH)
    Makie.save(joinpath(root_plots, "glaciers_H.pdf"), figH, pt_per_unit = 1)

end


function plot_avg_longterm_temps(climate, gdirs)
    mean_longterm_temps, labels = [],[]
    for (climate_glacier, gdir) in zip(climate, gdirs)
        push!(mean_longterm_temps, climate_glacier["longterm_temps"].temp.data)
        push!(labels, gdir.rgi_id)
    end
    display(Plots.plot(mean_longterm_temps, label=permutedims(labels), 
                        xlabel="Years", ylabel="Mean longterm air temperature (°C)", legend=:topright;
                        palette=palette(:tab10,15)))
end


# TODO: correctly retrieve the glacier coordinates to plot them in `imshow` as an extent
function plot_monthly_map(climate, variable, year)
    climate = climate[variable].where(climate.time.dt.year == year, drop=true).groupby("time.month")
    fig_clim, ax_clim = pplt.subplots([1:6, 7:12], axheight=2)
    fig_clim.format(
        abc=true, abcloc="ul", suptitle= ("$year - monthly $variable")
    )
    for mon in 1:12
        if variable == "temp"
            m_var = ax_clim[mon].imshow(climate.mean()[mon], cmap="Thermal", 
                    vmin=minimum(climate.mean().data), vmax=maximum(climate.mean().data)) # set common min max temp
        else
            m_var = ax_clim[mon].imshow(climate.sum()[mon], cmap="DryWet", 
                    vmin=climate.sum().min().data, vmax=climate.sum().max().data) # set common min max precipitation
        end
        ax_clim[mon].set_title(Dates.monthname(mon))
        if(mon == 12)
            if variable == "temp"
                fig_clim.colorbar(m_var, label="Air temperature (°C)")
            else
                fig_clim.colorbar(m_var, label="Accumulated $variable (mm)")
            end
        end
    end
end