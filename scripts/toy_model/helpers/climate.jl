using Flux

###############################################
############  FUNCTIONS   #####################
###############################################

# Get the corresponding climate dataset for each gdir
function get_gdirs_with_climate(gdirs, plot=true)
    climate_raw = get_climate(gdirs)
    climate = filter_climate(climate_raw) 
    if plot
        plot_avg_longterm_temps(climate, gdirs)
    end
    gdirs_climate = get_gdir_climate_tuple(gdirs, climate)
    return gdirs_climate
end

function get_MB_climate_datasets(gdir, mb, period_y, fs)
    # Retrieve reference MB data
    rgi_id = gdir.rgi_id
    println("Downloading climate data for $rgi_id...")
    climate = get_raw_climate_data(gdir)

    if mb
        println("Fetching mass balance reference data...")
        try
            # Building MB and climate datasets
            mb_glaciological = gdir.get_ref_mb_data(input_filesuffix=fs)
            # Get hydrological period matching the reference MB data
            hydro_period = to_hydro_period(mb_glaciological)
            climate_ds, MB_ds = build_MB_climate_ds(hydro_period, climate, gdir, mb_glaciological)
            return MB_ds, climate_ds   
        catch error
            @warn "$error: Error retrieving reference mass balance data. Retrieving only climate data."
            # Building only climate dataset
            hydro_period = collect(Date(period_y[1],10,1):Day(1):Date(period_y[2],09,30))
            climate_ds, MB_ds = build_climate_ds(hydro_period, climate, gdir)
            return MB_ds, climate_ds  
        end
    else
        # Building only climate dataset
        hydro_period = collect(Date(period_y[1],10,1):Day(1):Date(period_y[2],09,30))
        climate_ds, MB_ds = build_climate_ds(hydro_period, climate, gdir)
        return MB_ds, climate_ds
    end     
end

function build_MB_climate_ds(hydro_period, climate, gdir, mb_glaciological)
    println("Processing data for reference period ", hydro_period[1], " - ", hydro_period[end])
    # Build the training and reference dataset including annual and seasonal data
    # Get the climate forcings for the MB data
    seasons = ["annual", "accumulation", "ablation"]
    balances = ["ANNUAL_BALANCE", "WINTER_BALANCE", "SUMMER_BALANCE"]
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
    return climate_ds, MB_ds
end

function build_climate_ds(hydro_period, climate, gdir)
    ds_clim_buffer = []
    println("Processing data for reference period ", hydro_period[1], " - ", hydro_period[end])
    MB_clim = get_climate_forcing(gdir, climate, hydro_period, "annual")
    push!(ds_clim_buffer, MB_clim)
    push!(ds_clim_buffer, nothing)
    push!(ds_clim_buffer, nothing)
    # Store everything in a global dataset 
    println("Storing data in metadataset")
    climate_ds = Dataset(ds_clim_buffer...)
    MB_ds = nothing
    return climate_ds, MB_ds
end

# Compute climate forcings for a given period to feed MB model
function get_climate_forcing(gdir, climate, period, season)
    @assert any(season .== ["annual","accumulation", "ablation"]) "Wrong season type, must be `annual`, `accumulation` or `ablation`"

    # Make sure the desired period is covered by the climate data
    period = trim_period(period, climate) 
    @assert any((climate.time[1].dt.date.data[1] <= period[1]) & any(climate.time[end].dt.date.data[1] >= period[end])) "No overlapping period available between climate and MB data!" 
    climate = climate.sel(time=period) # Crop desired time period

    if season == "accumulation"
        climate = climate.sel(time=is_acc.(clim_period.time.dt.month.data))
    elseif season == "ablation"
        climate = climate.sel(time=is_abl.(clim_period.time.dt.month.data))
    end

    # Transform climate to cumulative PDDs, snowfall and rainfall
    climate = get_cumulative_climate(climate)

    # Convert climate dataset to 2D based on the glacier's DEM
    g_dem = xr.open_rasterio(gdir.get_filepath("dem"))
    climate = create_2D_climate_data(climate, g_dem)
        
    # Apply temperature gradients and compute snow/rain fraction for the selected period
    apply_t_grad!(climate, g_dem)
    climate = climate.drop("gradient") 

    return climate
     
end

function get_cumulative_climate(climate, gradient_bounds=[-0.009, -0.003], default_grad=-0.0065)
    avg_temp = climate.temp.resample(time="1M").mean() 
    avg_gradients = climate.gradient.resample(time="1M").mean() 
    climate.temp.data = climate.temp.where(climate.temp > 0, 0).data # get PDDs
    climate.gradient.data = utils.clip_array(climate.gradient.data, gradient_bounds[1], gradient_bounds[2]) # Clip gradients within plausible values
    attributes = climate.attrs
    climate_sum = climate.resample(time="1M").sum() # get monthly cumulative values
    climate_sum_avg1 = climate_sum.assign(Dict("avg_temp"=>avg_temp)) 
    climate_sum_avg2 = climate_sum_avg1.assign(Dict("avg_gradient"=>avg_gradients))
    climate_sum_avg2.attrs = attributes
    return climate_sum_avg2
end

function get_raw_climate_data(gdir, temp_resolution="daily", climate="W5E5")
    MBsandbox.process_w5e5_data(gdir, climate_type=climate, temporal_resol=temp_resolution) 
    fpath = gdir.get_filepath("climate_historical", filesuffix="_daily_W5E5")
    climate = xr.open_dataset(fpath)
    return climate
end

# Function to apply temperature lapse rates to the full matrix of a glacier
function apply_t_grad!(climate, g_dem)
    # We apply the gradients to the temperature
    climate.temp.data = climate.temp.data .+ climate.avg_gradient.data .* (g_dem.data .- climate.ref_hgt)
    climate.PDD.data = climate.PDD.data .+ climate.gradient.data .* (g_dem.data .- climate.ref_hgt)
    # We adjust the rain/snow fractions with the updated temperature
    climate.snow.data = climate.snow.where(climate.temp < 0, 0).data
    climate.rain.data = climate.rain.where(climate.temp > 0, 0).data
end

# Function to convert the baseline OGGM climate dataset to 2D
function create_2D_climate_data(climate, g_dem)
    # Create dummy 2D arrays to have a base to apply gradients afterwards
    dummy_grid = ones(size(permutedims(g_dem.data, (1,2,3))))
    temp_2D = climate.avg_temp.data .* dummy_grid
    PDD_2D = climate.temp.data .* dummy_grid
    snow_2D = climate.prcp.data .* dummy_grid
    rain_2D = climate.prcp.data .* dummy_grid

    # We generate a new dataset with the scaled data
    climate_2D = xr.Dataset(
        data_vars=Dict([
            ("temp", (["time","y","x"], temp_2D)),
            ("PDD", (["time","y","x"], PDD_2D)),
            ("snow", (["time","y","x"], snow_2D)),
            ("rain", (["time","y","x"], rain_2D)),
            ("gradient", (["time"], climate.gradient.data)),
            ("avg_gradient", (["time"], climate.avg_gradient.data))
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

function get_climate(gdirs)
    println("Getting climate data...")
    # Retrieve and compute climate data in parallel
    # /!\ Keep batch size small in order to avoid memory problems
    climate = pmap(gdir -> get_climate_glacier(gdir), gdirs; batch_size=10) 
    return climate
end

function get_climate_glacier(gdir, mb=true, period_y=(1979,2019), full=false)
    # Generate downscaled climate data
    if !isfile(joinpath(gdir.dir, "annual_temps.nc")) || overwrite_climate # Retrieve unless overwrite
        mb_type = "mb_real_daily"
        grad_type = "var_an_cycle" # could use here as well 'cte'
        # fs = "_daily_".*climate
        fs = "_daily_W5E5"
        MB_ds, climate_ds = get_MB_climate_datasets(gdir, mb, period_y, fs)
        # Convert to annual values to force UDE
        annual_temps = climate_ds.annual.temp.groupby("time.year").mean(dim=["time", "x", "y"])
        # yb,ye = annual_temps.year.data[1], annual_temps.year.data[end]
        println("Storing climate data in: ", joinpath(gdir.dir, "annual_temps.nc"))
        annual_temps.to_netcdf(joinpath(gdir.dir, "annual_temps.nc"))
        annual_temps = xr.open_dataset(joinpath(gdir.dir, "annual_temps.nc"))
    else
        annual_temps = xr.open_dataset(joinpath(gdir.dir, "annual_temps.nc"))
        MB_ds, climate_ds = nothing, nothing # dummy empty variables
    end
    # Compute a 20-year rolling mean for long-term air temperature variability
    longterm_temps = annual_temps.rolling(year=20).mean().dropna("year")
    # return longterm_temps, annual_temps, MB_ds, climate_ds
    climate = Dict("longterm_temps"=>longterm_temps, "annual_temps"=>annual_temps,
                    "MB_dataset"=>MB_ds, "climate_dataset"=>climate_ds)
    return climate
end

function filter_climate(climate)
    updated_climate = []
    for climate_glacier in climate
        if length(climate_glacier["longterm_temps"].temp.data) >= t₁ 
            push!(updated_climate, climate_glacier)
        else
            @warn "Filtered glacier due to short climate series"
        end
    end
    return updated_climate
end

function get_gdir_climate_tuple(gdirs, climate)
    dates, longterm_temps, annual_temps = [],[],[]
    for climate_batch in climate
        push!(dates, climate_batch["longterm_temps"].year.data)
        push!(longterm_temps, climate_batch["longterm_temps"].temp.data)
        push!(annual_temps, climate_batch["annual_temps"].temp.data)
    end
    gdirs_climate = (dates, gdirs, longterm_temps, annual_temps)
    return gdirs_climate
end


end # @everywhere

