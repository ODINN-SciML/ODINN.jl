using Infiltrator
export plot_avg_longterm_temps, plot_glacier_dataset

function plot_glacier_dataset(gdirs_climate, gdir_refs, random_MB; display=false)
    if plots[]
        println("Plotting glacier dataset...")
        # gdirs_climate = (dates, gdirs, longterm_temps, annual_temps)
        gdirs = gdirs_climate[2]
        gdirs_spinup = load(joinpath(ODINN.root_dir, "data/spinup/gdir_refs.jld2"))["gdir_refs"]
        theme = Theme(fontsize=18, font="TeX Gyre Heros")
        set_theme!(theme)

        names = ["Argentiere", "Peyto Glacier", "Edvardbreen", "Aletsch", "Lemon Creek Glacier", "Biskayerfonna",
                "Storglaciaren", "Wolverine Glacier", "Gulkana Glacier", "Esetuk Glacier", ]
        
        # Ice surface velocities
        figV = Makie.Figure(resolution=(1200, 1800))
        n = 5
        m = floor(Int, length(gdirs_climate[2])/n)
        # axsV = [GeoAxis(figV[i, j], source = "+proj=tmerc +lon_0=$(gdirs[k].cenlon) +lat_0=$(gdirs[k].cenlat)", dest = "+proj=tmerc +lon_0=12.5 +lat_0=42", 
        #                 lonlims=(gdirs[k].extent_ll[1,1], gdirs[k].extent_ll[1,2]), latlims=(gdirs[k].extent_ll[2,1], gdirs[k].extent_ll[2,2]), 
        #                 coastlines = true, remove_overlapping_ticks = true) 
        #                 for i in 1:n, j in 1:m, k in 1:length(gdirs)]

        axsV = []
        k = 1
        for i in 1:n
            for j in 1:m
                push!(axsV, GeoAxis(figV[i, j], source = "+proj=tmerc +lon_0=$(gdirs[k].cenlon) +lat_0=$(gdirs[k].cenlat)", 
                            dest = "+proj=tmerc +lon_0=$(gdirs[k].cenlon) +lat_0=$(gdirs[k].cenlat)", 
                            lonlims=(gdirs[k].extent_ll[1,1], gdirs[k].extent_ll[1,2]), 
                            latlims=(gdirs[k].extent_ll[2,1], gdirs[k].extent_ll[2,2]), 
                            coastlines = true, remove_overlapping_ticks = true))
                k += 1
            end
        end

        # hidedecorations!.(axsV)
        tightlimits!.(axsV)
        # datalims!.(axsV)
        Vs, hm_Vs = [],[]
        for (i, ax) in enumerate(axsV)
            ax.aspect = DataAspect()
            # glacier_gd = xr.open_dataset(gdirs[i].get_filepath("gridded_data"))
            # nx = glacier_gd.y.size # glacier extent
            # ny = glacier_gd.x.size # really weird, but this is inversed 
            # Δx = abs(gdirs[i].grid.dx)
            # Δy = abs(gdirs[i].grid.dy)
            rgi_id = gdirs[i].rgi_id
            name = gdirs[i].name
            ax.title = "$name - $rgi_id \n $(gdirs[i].cenlat)° - $(gdirs[i].cenlon)°"
            Vx = gdir_refs[i]["Vx"]
            Vy = gdir_refs[i]["Vy"]
            V = (abs.(Vx) .+ abs.(Vy))./2
            lons = gdirs[i].extent_ll[1,:]
            lats = gdirs[i].extent_ll[2,:]

            push!(Vs,V)
            lonrange = collect(range(lons[1], lons[2], length=size(V)[1]))
            latrange = collect(range(lats[1], lats[2], length=size(V)[2]))
            push!(hm_Vs, Makie.heatmap!(ax, lonrange, latrange, fillZeros(V), colormap=:speed))
        end
        minV = minimum(minimum.(Vs))
        maxV = maximum(maximum.(Vs)) 
        Colorbar(figV[2:3,m+1], limits=(minV,maxV), colormap=:speed, label="Ice surface velocity")
        Label(figV[0, :], text = "Glacier dataset", textsize = 30)
        if display
            display(figV)
        end
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
            if ice_thickness_source == "millan"
                H = Float32.(ifelse.(glacier_gd.glacier_mask.data .== 1, glacier_gd.millan_ice_thickness.data, 0.0))
            elseif ice_thickness_source == "farinotti"
                H = Float32.(glacier_gd.consensus_ice_thickness.data)
                # H = gdir_refs[i]["H"]
                fillZeros!(H)
            end
            smooth!(H)
            push!(Hs, fillNaN(H))
            push!(hm_Hs, Makie.heatmap!(ax, H, colormap=:ice))
        end
        minH = minimum(minimum.(Hs))
        maxH = maximum(maximum.(Hs)) 
        Colorbar(figH[2:3,m+1], limits=(minH,maxH), colormap=:ice, label="Ice thickness (m)")
        Label(figH[0, :], text = "Glacier dataset", textsize = 30)
        if display
            display(figH)
        end
        Makie.save(joinpath(root_plots, "glaciers_H.pdf"), figH, pt_per_unit = 1)

        # Surface elevation difference
        figS = Makie.Figure(resolution=(1200, 1800))
        axsS = [Axis(figS[i, j]) for i in 1:n, j in 1:m]
        hidedecorations!.(axsS)
        tightlimits!.(axsS)
        Ss, S_diffs, hm_Ss = [],[],[]
        for (i, ax) in enumerate(axsS)
            ax.aspect = DataAspect()
            glacier_gd = xr.open_dataset(gdirs[i].get_filepath("gridded_data"))
            
            nx = glacier_gd.y.size # glacier extent
            ny = glacier_gd.x.size # really weird, but this is inversed 
            Δx = abs(gdirs[i].grid.dx)
            Δy = abs(gdirs[i].grid.dy)
            rgi_id = gdirs[i].rgi_id
            name = gdirs[i].name
            ax.title = "$name - $rgi_id"
            S = gdir_refs[i]["S"] # final surface elevation
            H = gdir_refs[i]["H"]
            fillNaN!(H)

            # We get the spinup gdirs data
            if use_spinup[]
                H = gdirs_spinup[i]["H"]
                B = gdirs_spinup[i]["B"]
            else
                if ice_thickness_source == "millan"
                    H₀ = Float32.(ifelse.(glacier_gd.glacier_mask.data .== 1, glacier_gd.millan_ice_thickness.data, 0.0))
                    fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
                elseif ice_thickness_source == "farinotti"
                    H₀ = Float32.(glacier_gd.consensus_ice_thickness.data)
                    fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
                end
                smooth!(H)
                B = gdir_refs[i]["B"] 
            end
            S₀ = B .+ H₀
            S_diff = S .- S₀

            push!(Ss, fillNaN(S_diff))
            push!(S_diffs, fillZeros(S_diff))
        end
        minS = minimum(minimum.(Ss))
        maxS = maximum(maximum.(Ss)) 
        # absmaxS = maximum([abs(minS), abs(maxS)])
        absmaxS = 50.0f0

        # Make heatmap and colorbars with common S limits
        for (i, ax) in enumerate(axsS)
            push!(hm_Ss, Makie.heatmap!(ax, S_diffs[i], colormap=Reverse(:balance), colorrange=(-absmaxS,absmaxS), clim=(-absmaxS,absmaxS)))
        end
        Colorbar(figS[2:3,m+1], colorrange=(-absmaxS,absmaxS),colormap=Reverse(:balance), label="Surface altitude difference (m)")
        Label(figS[0, :], text = "Glacier dataset", textsize = 30)
        if display 
            display(figS)
        end
        Makie.save(joinpath(root_plots, "glaciers_S.pdf"), figS, pt_per_unit = 1)

        # Surface mass balance
        figMB = Makie.Figure(resolution=(1200, 1800))
        axsMB = [Axis(figMB[i, j]) for i in 1:n, j in 1:m]
        hidedecorations!.(axsMB)
        tightlimits!.(axsMB)
        MBs, MBs_nan, hm_MBs = [],[],[]
        for (i, ax) in enumerate(axsMB)
            ax.aspect = DataAspect()
            glacier_gd = xr.open_dataset(gdirs[i].get_filepath("gridded_data"))
            
            nx = glacier_gd.y.size # glacier extent
            ny = glacier_gd.x.size # really weird, but this is inversed 
            Δx = abs(gdirs[i].grid.dx)
            Δy = abs(gdirs[i].grid.dy)
            rgi_id = gdirs[i].rgi_id
            name = gdirs[i].name
            ax.title = "$name - $rgi_id"
            B = gdir_refs[i]["B"]
            H = glacier_gd.consensus_ice_thickness.data # initial ice thickness conditions for forward model
            fillNaN!(H)
            smooth!(H)
            S = B .+ H

            MB = compute_MB_matrix(random_MB[i], S, H, 2)

            push!(MBs, fillNaN(MB))
            push!(MBs_nan, MB)
            
        end
        minS = minimum(minimum.(MBs))
        maxS = maximum(maximum.(MBs)) 
        absmaxMB = maximum([abs(minS), abs(maxS)])
        # absmaxS = 50.0f0

        # Make heatmap and colorbars with common S limits
        for (i, ax) in enumerate(axsMB)
            push!(hm_MBs, Makie.heatmap!(ax, MBs_nan[i], colormap=Reverse(:balance), colorrange=(-absmaxMB,absmaxMB), clim=(-absmaxMB,absmaxMB)))
        end
        Colorbar(figMB[2:3,m+1], colorrange=(-absmaxMB,absmaxMB),colormap=Reverse(:balance), label="Surface Mass Balance (m.w.e.)")
        Label(figMB[0, :], text = "Glacier dataset", textsize = 30)
        if display 
            display(figMB)
        end
        Makie.save(joinpath(root_plots, "glaciers_MB.pdf"), figMB, pt_per_unit = 1)


        println("Glacier dataset plots stored")
    end
end


function plot_avg_longterm_temps(climate, gdirs)
    if plots
        mean_longterm_temps, labels = [],[]
        for (climate_glacier, gdir) in zip(climate, gdirs)
            push!(mean_longterm_temps, climate_glacier["longterm_temps"].temp.data)
            push!(labels, gdir.rgi_id)
        end
        display(Plots.plot(mean_longterm_temps, label=permutedims(labels), 
                            xlabel="Years", ylabel="Mean longterm air temperature (°C)", legend=:topright;
                            palette=palette(:tab10,15)))
    end
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