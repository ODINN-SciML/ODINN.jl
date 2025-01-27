export generate_random_MB, apply_MB_mask!

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