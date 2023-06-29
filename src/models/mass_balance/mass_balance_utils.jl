
function compute_MB(mb_model::TImodel1, climate_2D_period::Climate2Dstep)
    return ((mb_model.acc_factor .* climate_2D_period.snow) .- (mb_model.DDF .* climate_2D_period.PDD))
end

function MB_timestep(model::Model, glacier::Glacier, params_solver::SolverParameters, t::F) where {F <: AbstractFloat}
    # First we get the dates of the current time and the previous step
    period = partial_year(Day, t - params_solver.step):Day(1):partial_year(Day, t)
    climate_step::PyObject = get_cumulative_climate(glacier.climate.sel(time=period))
    # Convert climate dataset to 2D based on the glacier's DEM
    climate_2D_step::PyObject = downscale_2D_climate(climate_step, glacier)
    MB::Matrix{F} = compute_MB(model.mb_model, climate_2D_step)
    return MB
end

         
function MB_timestep!(model::Model, glacier::Glacier, params_solver::SolverParameters, t::F) where {F <: AbstractFloat}
    # First we get the dates of the current time and the previous step
    period = partial_year(Day, t - params_solver.step):Day(1):partial_year(Day, t)
    @timeit get_timer("ODINN") "Cumulative climate" begin
    get_cumulative_climate!(glacier.climate, period)
    end
    # Convert climate dataset to 2D based on the glacier's DEM
    @timeit get_timer("ODINN") "Downscale 2D climate" begin
    downscale_2D_climate!(glacier)
    end
    @timeit get_timer("ODINN") "Compute MB" begin
    model.iceflow.MB .= compute_MB(model.mass_balance, glacier.climate.climate_2D_step)
    end
end

function apply_MB_mask!(H::Matrix{F}, glacier::Glacier, ifm::IceflowModel) where {F <: AbstractFloat}
    # Appy MB only over ice, and avoid applying it to the borders in the accummulation area to avoid overflow
    MB::Matrix{F}, MB_mask::BitMatrix, MB_total::Matrix{F} = ifm.MB, ifm.MB_mask, ifm.MB_total
    MB_mask .= ((H .> 0.0) .&& (MB .< 0.0)) .|| ((H .> 0.0) .&& (glacier.dist_border .> 1.0) .&& (MB .>= 0.0)) 
    H[MB_mask] .+= MB[MB_mask]
    MB_total[MB_mask] .+= MB[MB_mask]
end

# function apply_MB_mask!(H, MB, MB_total, dist_border::Matrix{Float64})
#     # Appy MB only over ice, and avoid applying it to the borders in the accummulation area to avoid overflow
#     MB_mask = ((H .> 0.0) .&& (MB .< 0.0)) .|| ((H .> 0.0) .&& (dist_border .> 1.0) .&& (MB .>= 0.0))
#     H[MB_mask] .+= MB[MB_mask]
#     MB_total[MB_mask] .+= MB[MB_mask]
# end