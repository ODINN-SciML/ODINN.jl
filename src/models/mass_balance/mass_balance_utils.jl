
function compute_MB(mb_model::TImodel1, climate_2D_period::PyObject)
    return (mb_model.acc_factor .* climate_2D_period.snow.data) .- (mb_model.DDF .* climate_2D_period.PDD.data)
end

function MB_timestep(mb_model::MBmodel, climate::Climate, S, S_coords, t, step)
    # First we get the dates of the current time and the previous step
    period = partial_year(Day, t - step):Day(1):partial_year(Day, t)
    climate_step = get_cumulative_climate(climate.sel(time=period))
    # Convert climate dataset to 2D based on the glacier's DEM
    climate_2D_step = downscale_2D_climate(climate_step, S, S_coords)
    MB = compute_MB(mb_model, climate_2D_step)
    return MB
end

function MB_timestep!(MB, mb_model::MBmodel, climate, S, S_coords, t, step)
    # First we get the dates of the current time and the previous step
    period = partial_year(Day, t - step):Day(1):partial_year(Day, t)
    get_cumulative_climate!(climate, period)
    # Convert climate dataset to 2D based on the glacier's DEM
    downscale_2D_climate!(climate, S, S_coords)
    MB .= compute_MB(mb_model, climate.climate_2D_step[])
end

function apply_MB_mask!(H, MB, MB_total, context::Tuple)
    dist_border = context[33]
    #slope = context[34]
    MB_mask = context[35]
    # Appy MB only over ice, and avoid applying it to the borders in the accummulation area to avoid overflow
    MB_mask .= ((H .> 0.0) .&& (MB .< 0.0)) .|| ((H .> 0.0) .&& (dist_border .> 1.0) .&& (MB .>= 0.0))
    H[MB_mask] .+= MB[MB_mask]
    MB_total[MB_mask] .+= MB[MB_mask]
end

function apply_MB_mask!(H, MB, MB_total, dist_border::Matrix{Float64})
    # Appy MB only over ice, and avoid applying it to the borders in the accummulation area to avoid overflow
    MB_mask = ((H .> 0.0) .&& (MB .< 0.0)) .|| ((H .> 0.0) .&& (dist_border .> 1.0) .&& (MB .>= 0.0))
    H[MB_mask] .+= MB[MB_mask]
    MB_total[MB_mask] .+= MB[MB_mask]
end