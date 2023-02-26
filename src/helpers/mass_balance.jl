export generate_random_MB

###############################################
############  FUNCTIONS   #####################
###############################################

# function generate_random_MB(gdirs_climate, tspan; plot=true)
#     random_MB = [] # tuple with RGI_ID, MB_max, MB_min
#     gdirs = gdirs_climate[2]
#     climates = gdirs_climate[3]
#     for (gdir, climate) in zip(gdirs, climates)
#         scaling = 1.0
#         clim = scaling*(1.0/abs(mean(climate)))^(1.0/7.0) # climate driver to adapt random MB
#         # clim = (clim <= 1.1)*clim # clip large values
#         MB_max = (ref_max_MB .+ randn(MersenneTwister(1),floor(Int,tspan[2]-tspan[1]+1))).*clim
#         MB_min = (ref_min_MB .+ randn(MersenneTwister(2),floor(Int,tspan[2]-tspan[1]+1))).*clim
#         # MB_min = ifelse.(MB_min.>=-15.0, MB_min, -15.0)
#         push!(random_MB, (gdir.rgi_id, MB_max, MB_min))
#     end

#     if plot
#         MBmax_series, MBmin_series, labels = [],[],[]
#         for glacier_MB in random_MB
#             push!(MBmax_series, glacier_MB[2])
#             push!(MBmin_series, glacier_MB[3])
#             push!(labels, glacier_MB[1])
#         end
#         Plots.plot(MBmax_series, 
#                 xlabel="Years", ylabel="Max/min mass balance (m.w.e./yr)", label="",
#                 legend=:topright;
#                 palette=palette(:blues,15))
#         MBplot = Plots.plot!(MBmin_series, label="";
#                 palette=palette(:reds,15))
#         display(MBplot)
#         Plots.savefig(MBplot,joinpath(root_plots,"MBseries.png"))
#         Plots.savefig(MBplot,joinpath(root_plots,"MBseries.pdf"))
#     end

#     return random_MB
# end

# function compute_MB_matrix!(context, H, year)
#     # MB array has tuples with (RGI_ID, MB_max, MB_min)
#     B = context[2]
#     MB_series = context[24]
#     simulation_years = context[31]
#     max_MB = MB_series[2][year .== simulation_years]
#     min_MB = MB_series[3][year .== simulation_years]
    
#     # Add mass balance based on gradient
#     max_S = context[28]
#     min_S = context[29] 
#     max_S .= maximum(getindex(B, H .> 0.0) .+ getindex(H, H .> 0.0))
#     min_S .= minimum(getindex(B, H .> 0.0) .+ getindex(H, H .> 0.0))

#     # Define the mass balance as line between minimum and maximum surface
#     MB = context[25]
#     MB .= (min_MB .+ (B .+ H .- min_S) .* 
#                 ((max_MB .- min_MB) ./ (max_S .- min_S)) .* Matrix(H.>0.0)) ./ 12.0 # TODO: control MB timestepping
# end

# function compute_MB_matrix(context, H, year)
#     S = context[1]
#     simulation_years = context[12]
#     max_MB = context[7][1][2][year .== simulation_years]
#     min_MB = context[7][1][3][year .== simulation_years]
#     max_S = maximum(getindex(S, H .> 0.0))
#     min_S = minimum(getindex(S, H .> 0.0))
#     MB = (min_MB .+ (S .- min_S) .* ((max_MB .- min_MB) ./ (max_S .- min_S))) .* Matrix{Float64}(H.>0.0) ./ 12.0 # TODO: control MB timestepping
#     return MB
# end

# function compute_MB_matrix(random_MB::Tuple{String, Vector{Float64}, Vector{Float64}}, S, H, year)
#     max_MB = random_MB[2][year]
#     min_MB = random_MB[3][year]
#     max_S = maximum(getindex(S, H .> 0.0))
#     min_S = minimum(getindex(S, H .> 0.0))
#     MB = (min_MB .+ (S .- min_S) .* ((max_MB - min_MB) / (max_S - min_S))) .* Matrix{Float64}(H.>0.0)
#     return MB
# end

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
