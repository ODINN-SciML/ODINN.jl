export generate_random_MB

###############################################
############  DATA STRUCTURES #################
###############################################

### Data structures
# Abstract type as a parent type for Mass Balance models
abstract type MB_model end
# Subtype structure for Temperature-Index Mass Balance model
abstract type TI_model <: MB_model end
# Temperature-index model with 1 melt factor
# Make these mutable if necessary
@kwdef struct TI_model_1 <: TI_model
    DDF::Float64
    acc_factor::Float64
end

@kwdef struct TI_model_2 <: TI_model
    DDF_snow::Float64
    DDF_ice::Float64
    acc_factor::Float64
end

###############################################
############  FUNCTIONS   #####################
###############################################

function compute_MB(mb_model::TI_model_1, climate_2D_period::PyObject)
    return (mb_model.acc_factor .* climate_2D_period.snow.data) .- (mb_model.DDF .* climate_2D_period.PDD.data)
end

function MB_timestep(mb_model::MB_model, climate, S, S_coords, t, step)
    # First we get the dates of the current time and the previous step
    period = partial_year(Day, t - step):Day(1):partial_year(Day, t)
    climate_step = get_cumulative_climate(climate.sel(time=period))
    # Convert climate dataset to 2D based on the glacier's DEM
    climate_2D_step = downscale_2D_climate(climate_step, S, S_coords)
    MB = compute_MB(mb_model, climate_2D_step)
    return MB
end

function MB_timestep!(MB, mb_model::MB_model, climate, S, S_coords, t, step)
    # First we get the dates of the current time and the previous step
    period = partial_year(Day, t - step):Day(1):partial_year(Day, t)
    @timeit to "Climate step" begin
    get_cumulative_climate!(climate, period)
    end
    # Convert climate dataset to 2D based on the glacier's DEM
    @timeit to "Climate 2D step" begin
    downscale_2D_climate!(climate, S, S_coords)
    end
    @timeit to "Compute MB" begin
    MB .= compute_MB(mb_model, climate.climate_2D_step[])
    end
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