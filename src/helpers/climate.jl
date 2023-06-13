###############################################
############  FUNCTIONS   #####################
###############################################

using Dates # to provide correct Julian time slices 

function generate_raw_climate_files(gdir, tspan)
    if !ispath(joinpath(gdir.dir, "raw_climate_$tspan.nc"))
        println("Getting raw climate data for: ", gdir.rgi_id)
        # Get raw climate data for gdir
        tspan_date = partial_year(Day, tspan[1]):Day(1):partial_year(Day, tspan[2])
        climate =  get_raw_climate_data(gdir)
        # Make sure the desired period is covered by the climate data
        period = trim_period(tspan_date, climate) 
        if any((climate.time[1].dt.date.data[1] <= period[1]) & any(climate.time[end].dt.date.data[1] >= period[end]))
            climate = climate.sel(time=period) # Crop desired time period
        else
            @warn "No overlapping period available between climate tspan!" 
        end
        # Save raw gdir climate on disk 
        climate.to_netcdf(joinpath(gdir.dir, "raw_climate_$tspan.nc"))
        GC.gc()
    end
end


"""
    get_cumulative_climate(climate, gradient_bounds=[-0.009, -0.003], default_grad=-0.0065)

Computes Positive Degree Days (PDDs) and cumulative rainfall and snowfall from climate data.
"""
function get_cumulative_climate!(climate, period, gradient_bounds=[-0.009, -0.003], default_grad=-0.0065)
    climate.climate_raw_step[] = climate.raw_climate.sel(time=period)
    climate.avg_temps[] = climate.climate_raw_step[].temp.mean() 

    climate.avg_gradients[] = climate.climate_raw_step[].gradient.mean() 
    climate.climate_raw_step[].temp.data = climate.climate_raw_step[].temp.where(climate.climate_raw_step[].temp > 0, 0).data # get PDDs
    climate.climate_raw_step[].gradient.data = utils.clip_array(climate.climate_raw_step[].gradient.data, gradient_bounds[1], gradient_bounds[2]) # Clip gradients within plausible values
    climate.climate_step[] = climate.climate_raw_step[].sum() # get monthly cumulative values
    climate.climate_step[] = climate.climate_step[].assign(Dict("avg_temp"=>climate.avg_temps[])) 
    climate.climate_step[] = climate.climate_step[].assign(Dict("avg_gradient"=>climate.avg_gradients[]))
    climate.climate_step[].attrs = climate.climate_raw_step[].attrs
end

function get_cumulative_climate(climate, gradient_bounds=[-0.009, -0.003], default_grad=-0.0065)
    avg_temp = climate.temp.mean() 
    avg_gradients = climate.gradient.mean() 
    climate.temp.data = climate.temp.where(climate.temp > 0, 0).data # get PDDs
    climate.gradient.data = utils.clip_array(climate.gradient.data, gradient_bounds[1], gradient_bounds[2]) # Clip gradients within plausible values
    attributes = climate.attrs
    climate_sum = climate.sum() # get monthly cumulative values
    climate_sum = climate_sum.assign(Dict("avg_temp"=>avg_temp)) 
    climate_sum = climate_sum.assign(Dict("avg_gradient"=>avg_gradients))
    climate_sum.attrs = attributes
    return climate_sum
end

"""
    get_raw_climate_data(gdir, temp_resolution="daily", climate="W5E5")

Downloads the raw W5E5 climate data with a given resolution (daily by default). Returns an xarray Dataset. 
"""
function get_raw_climate_data(gdir; temp_resolution="daily", climate="W5E5")
    MBsandbox.process_w5e5_data(gdir, climate_type=climate, temporal_resol=temp_resolution) 
    fpath = gdir.get_filepath("climate_historical", filesuffix="_daily_W5E5")
    climate = xr.open_dataset(fpath)
    return climate
end

"""
    apply_t_grad!(climate, g_dem)

Applies temperature gradients to the glacier 2D climate data based on a DEM.  
"""
function apply_t_cumul_grad!(climate, S)
    # We apply the gradients to the temperature
    # /!\ AVOID USING `.` IN JULIA TO ASSIGN. IT'S NOT HANDLED BY XARRAY. USE `=` INSTEAD
    climate.temp.data = climate.temp.data .+ climate.avg_gradient.data .* (S .- climate.ref_hgt)
    climate.PDD.data = climate.PDD.data .+ climate.gradient.data .* (S .- climate.ref_hgt)
    climate.PDD.data = ifelse.(climate.PDD.data .< 0.0, 0.0, climate.PDD.data) # Crop negative PDD values

    # We adjust the rain/snow fractions with the updated temperature
    climate.snow.data = climate.snow.where(climate.temp < 0.0, 0.0).data
    climate.rain.data = climate.rain.where(climate.temp > 0.0, 0.0).data
end

function apply_t_grad!(climate, dem)
    # We apply the gradients to the temperature
    # /!\ AVOID USING `.` IN JULIA TO ASSIGN. IT'S NOT HANDLED BY XARRAY. USE `=` INSTEAD
    climate.temp.data = climate.temp.data .+ climate.gradient.data .* (mean(dem.data) .- climate.ref_hgt)
end

"""
    downscale_2D_climate(climate, g_dem)

Projects climate data to the glacier matrix by simply copying the closest gridpoint to all matrix gridpoints.
Generates a new xarray Dataset which is returned.   
"""
function downscale_2D_climate!(climate, S, S_coords)
    # Update 2D climate structure
    # /!\ AVOID USING `.` IN JULIA TO ASSIGN. IT'S NOT HANDLED BY XARRAY. USE `=` INSTEAD
    climate.climate_2D_step[].temp.data = climate.climate_step[].avg_temp.data .* ones(size(climate.climate_2D_step[].temp.data))
    climate.climate_2D_step[].PDD.data = climate.climate_step[].temp.data .* ones(size(climate.climate_2D_step[].PDD.data))
    climate.climate_2D_step[].snow.data = climate.climate_step[].prcp.data .* ones(size(climate.climate_2D_step[].snow.data))
    climate.climate_2D_step[].rain.data = climate.climate_step[].prcp.data .* ones(size(climate.climate_2D_step[].rain.data))
    # Update gradients
    climate.climate_2D_step[].gradient.data = climate.climate_step[].gradient.data
    climate.climate_2D_step[].avg_gradient.data = climate.climate_step[].avg_gradient.data

    # Apply temperature gradients and compute snow/rain fraction for the selected period
    apply_t_cumul_grad!(climate.climate_2D_step[], reshape(S, size(S))) # Reproject current S with xarray structure
end

function downscale_2D_climate(climate, S, S_coords)
    # Create dummy 2D arrays to have a base to apply gradients afterwards
    dummy_grid = ones(size(S))
    temp_2D = climate.avg_temp.data .* dummy_grid
    PDD_2D = climate.temp.data .* dummy_grid
    snow_2D = climate.prcp.data .* dummy_grid
    rain_2D = climate.prcp.data .* dummy_grid

    # We generate a new dataset with the scaled data
    climate_2D = xr.Dataset(
        data_vars=Dict([
            ("temp", (["y","x"], temp_2D)),
            ("PDD", (["y","x"], PDD_2D)),
            ("snow", (["y","x"], snow_2D)),
            ("rain", (["y","x"], rain_2D)),
            ("gradient", climate.gradient.data),
            ("avg_gradient", climate.avg_gradient.data)
            ]),
        coords=Dict([
            ("x", S_coords.x),
            ("y", S_coords.y)
        ]),
        attrs=climate.attrs
    )

    # Apply temperature gradients and compute snow/rain fraction for the selected period
    apply_t_cumul_grad!(climate_2D, reshape(S, size(S))) # Reproject current S with xarray structure
    return climate_2D

end

"""
    trim_period(period, climate)

Trims a time period based on the time range of a climate series. 
"""
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

function partial_year(period::Type{<:Period}, float)
    _year, Δ = divrem(float, 1)
    year_start = Date(_year)
    year = period((year_start + Year(1)) - year_start)
    partial = period(round(Dates.value(year) * Δ))
    year_start + partial
end
partial_year(float::AbstractFloat) = partial_year(Day, float)


function get_longterm_temps(gdir::PyObject, tspan)
    climate = xr.open_dataset(joinpath(gdir.dir, "raw_climate_$tspan.nc")) # load only once at the beginning
    dem = rioxarray.open_rasterio(gdir.get_filepath("dem"))
    apply_t_grad!(climate, dem)
    longterm_temps = climate.groupby("time.year").mean().temp.data
    return longterm_temps
end

function get_longterm_temps(gdir::PyObject, climate::PyObject)
    dem = rioxarray.open_rasterio(gdir.get_filepath("dem"))
    apply_t_grad!(climate, dem)
    longterm_temps = climate.groupby("time.year").mean().temp.data
    return longterm_temps
end

### Data structures
@kwdef mutable struct ClimateDataset
    raw_climate::PyObject # Raw climate dataset for the whole simulation
    # Buffers to avoid memory allocations
    climate_raw_step::Ref{PyObject} # Raw climate trimmed for the current step
    #climate_cum_step::Ref{PyObject} # Raw cumulative trimmed climate for the current step
    climate_step::Ref{PyObject} # Climate data for the current step
    climate_2D_step::Ref{PyObject} # 2D climate data for the current step to feed to the MB model
    longterm_temps::Vector{Float64} # Longterm temperatures for the ice rheology
    avg_temps::Ref{PyObject} # Intermediate buffer for computing average temperatures
    avg_gradients::Ref{PyObject} # Intermediate buffer for computing average gradients
end

function init_climate(gdir::PyObject, tspan, step, S, S_coords::PyObject)
    dummy_period = partial_year(Day, tspan[1]):Day(1):partial_year(Day, tspan[1] + step)
    raw_climate::PyObject = xr.open_dataset(joinpath(gdir.dir, "raw_climate_$tspan.nc"))
    climate_step = Ref{PyObject}(get_cumulative_climate(raw_climate.sel(time=dummy_period)))
    climate_2D_step = Ref{PyObject}(downscale_2D_climate(climate_step[], S, S_coords))
    longterm_temps = get_longterm_temps(gdir, raw_climate)
    climate = ClimateDataset(raw_climate = raw_climate,
                            climate_raw_step = raw_climate.sel(time=dummy_period),
                            #climate_cum_step = raw_climate.sel(time=dummy_period).sum(),
                            climate_step = climate_step,
                            climate_2D_step = climate_2D_step,
                            longterm_temps = longterm_temps,
                            avg_temps = raw_climate.sel(time=dummy_period).temp.mean(),
                            avg_gradients = raw_climate.sel(time=dummy_period).gradient.mean())
    return climate
end
