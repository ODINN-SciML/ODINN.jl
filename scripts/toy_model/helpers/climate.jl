using Flux

###############################################
############  FUNCTIONS   #####################
###############################################

function gen_MB_train_dataset(gdir, fs)
    # Retrieve reference MB data
    println("Downloading climate data for glacier...")
    climate = get_raw_climate_data(gdir)

    println("Fetching mass balance reference data...")
    mb_glaciological = gdir.get_ref_mb_data(input_filesuffix=fs)
    # Get hydrological period matching the reference MB data
    hydro_period = to_hydro_period(mb_glaciological)

    println("Processing data for reference period ", hydro_period[1], " - ", hydro_period[end])
    # Get the climate forcings for the MB data
    seasons = ["annual", "accumulation", "ablation"]
    balances = ["ANNUAL_BALANCE", "WINTER_BALANCE", "SUMMER_BALANCE"]

    # Build the training and reference dataset including annual and seasonal data
    ds_clim_buffer, ds_MB_buffer = [],[]
    for (season, balance) in zip(seasons, balances)
        season_MB = mb_glaciological[balance]
        MB_clim = get_climate_forcing(gdir, climate, hydro_period, season)
        push!(ds_MB_buffer, season_MB) 
        push!(ds_clim_buffer, MB_clim)
    end

    # Store everything in a global dataset 
    println("Storing data in metadataset")
    MB_ds = Dataset(ds_MB_buffer...) # splat it!!!
    climate_ds = Dataset(ds_clim_buffer...)

    return MB_ds, climate_ds        
end

# Compute climate forcings for a given period to feed MB model
function get_climate_forcing(gdir, climate, period, season)
    @assert any(season .== ["annual","accumulation", "ablation"]) "Wrong season type, must be `annual`, `accumulation` or `ablation`"

    # Make sure the desired period is covered by the climate data
    period = trim_period(period, climate) 
    @assert any((climate.time[1].dt.date.data[1] <= period[1]) & any(climate.time[end].dt.date.data[1] >= period[end])) "No overlapping period available between climate and MB data!" 
    clim_period = climate.sel(time=period) # Crop desired time period

    if season == "accumulation"
        clim_period = clim_period.sel(time=is_acc.(clim_period.time.dt.month.data))
    elseif season == "ablation"
        clim_period = clim_period.sel(time=is_abl.(clim_period.time.dt.month.data))
    end
        
    # Get glacier DEM
    g_dem = xr.open_rasterio(gdir.get_filepath("dem"))
    # Apply temperature gradients and compute snow/rain fraction for the selected period
    apply_t_grad!(clim_period, g_dem)
    # Obtained PDDs and cumulative snowfall and rain
    # clim_period.temp.data = clim_period.temp.where(clim_period.temp.data .> 0, 0) # PDDs
    clim_period = clim_period.drop("gradient") #.sum() # Accumulate everything

    return clim_period
     
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

function get_raw_climate_data(gdir, temp_resolution="daily", climate="W5E5", dim="2D")
    @assert any(dim .== ["1D", "2D"]) "Wrong number of dimensions $dim !. Needs to be either `1D` or `2D`."
    PARAMS["hydro_month_nh"]=1
    MBsandbox.process_w5e5_data(gdir, climate_type=climate, temporal_resol=temp_resolution) 
    fpath = gdir.get_filepath("climate_historical", filesuffix="_daily_W5E5")
    climate = xr.open_dataset(fpath)

    if dim == "2D"
        # Convert climate dataset to 2D based on the glacier's DEM
        g_dem = xr.open_rasterio(gdir.get_filepath("dem"))
        climate = create_2D_climate_data(climate, g_dem)
    end

    return climate
end

# Function to apply temperature lapse rates to the full matrix of a glacier
function apply_t_grad!(climate, g_dem, gradient_bounds=[-0.009, -0.003], default_grad=-0.0065)
    gradients = utils.clip_array(climate.gradient, gradient_bounds[1], gradient_bounds[2]) # Clip gradients within plausible values

    # We apply the gradients to the temperature
    climate.temp.data = climate.temp.data .+ gradients.data .* (g_dem.data .- climate.ref_hgt)
    # We adjust the rain/snow fractions with the updated temperature
    climate.snow.data = climate.snow.where(climate.temp < 0, 0).data
    climate.rain.data = climate.rain.where(climate.temp > 0, 0).data
end

# Function to convert the baseline OGGM climate dataset to 2D
function create_2D_climate_data(climate, g_dem)
    # Create dummy 2D arrays to have a base to apply gradients afterwards
    temp_2D = climate.temp.data .* ones(size(permutedims(g_dem.data, (1,2,3))))
    snow_2D = climate.prcp.data .* ones(size(permutedims(g_dem.data, (1,2,3))))
    rain_2D = climate.prcp.data .* ones(size(permutedims(g_dem.data, (1,2,3))))

    # We generate a new dataset with the scaled data
    climate_2D = xr.Dataset(
        data_vars=Dict([
            ("temp", (["time","y","x"], temp_2D)),
            ("snow", (["time","y","x"], snow_2D)),
            ("rain", (["time","y","x"], rain_2D)),
            ("gradient", (["time"], climate.gradient.data))
            ]),
        coords=Dict([
            ("time", climate.time),
            ("x", g_dem.x),
            ("y", g_dem.y)
        ]),
        attrs=climate.attrs
    )

    return climate_2D

end

function to_hydro_period(mass_balance::PyObject, trim_edges=true)
    # Select hydro period between October 1st and 30th of September
    if trim_edges
        hydro_period = collect(Date(mass_balance.index[1],10,1):Day(1):Date(mass_balance.index[end]-1,09,30))
    else
        hydro_period = collect(Date(mass_balance.index[1]-1,10,1):Day(1):Date(mass_balance.index[end],09,30))
    end

    return hydro_period
end

function to_hydro_period(years::Array)
    # Select hydro period between October 1st and 30th of September
    hydro_period = collect(Date(years[1]-1,10,1):Day(1):Date(years[end],09,30))

    return hydro_period
end

function trim_period(period, climate)
    if any(climate.time[1].dt.date.data[1] > period[1])
        head = jldate(climate.time[1])
        period = Date(year(head), 10, 1):Day(1):period[end] # make it a hydrological year
    end
    if any(climate.time[end].dt.date.data[1] < period[end])
        tail = jldate(climate.time[end])
        period = period[1]:Day(1):Date(year(tail), 9, 30) # make it a hydrological year
    end

    return period
end

function is_abl(month)
    return (month >= 4) & (month <= 10)
end

function is_acc(month)
    return (month <= 4) | (month >= 10)
end


# Metastructure to store xarray dataset with MB and climate data for training
struct Dataset
    annual
    accumulation
    ablation
end

@everywhere begin

function fake_temp_series(t, means=[0,-2.0,-3.0,-5.0,-10.0,-12.0,-14.0,-15.0,-20.0])
    temps, norm_temps, norm_temps_flat = [],[],[]
    for mean in means
       push!(temps, mean .+ rand(t).*1e-1) # static
       append!(norm_temps_flat, mean .+ rand(t).*1e-1) # static
    #    push!(temps, (mean .+ rand(t)) .+ 0.5.*ts) # with trend
    #    push!(temps, (mean .+ rand(t)) .+ -0.5.*ts) # with trend
    end

    # Normalise temperature series
    norm_temps_flat = Flux.normalise([norm_temps_flat...]) # requires splatting

    # Re-create array of arrays 
    for i in 1:t₁:length(norm_temps_flat)
        push!(norm_temps, norm_temps_flat[i:i+(t₁-1)])
    end

    return temps, norm_temps
end

end # @everywhere


