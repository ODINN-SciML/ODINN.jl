export Inversion

mutable struct InversionMetrics{F <: Real}
    rgi_id::Union{String, Nothing}
    A::F
    n::F
    C::Matrix{F}
    H_pred::Matrix{F}
    H_obs::Matrix{F}
    H_diff::Matrix{F} 
    V_pred::Matrix{F}
    V_obs::Matrix{F}
    V_diff::Matrix{F} 
    MSE::F
    Δx::F             
    Δy::F  
    lon::F 
    lat::F 
end

mutable struct Inversion  <: Simulation 
    model::Sleipnir.Model
    glaciers::Vector{Sleipnir.AbstractGlacier}
    parameters::Sleipnir.Parameters
    inversion::Vector{InversionMetrics}
end

"""
    function Inversion(
        model::Sleipnir.Model,
        glaciers::Vector{Sleipnir.AbstractGlacier},
        parameters::Sleipnir.Parameters
        )
Construnctor for Inversion struct with glacier model infomation, glaciers and parameters.
Keyword arguments
=================
"""
function Inversion(
    model::Sleipnir.Model,
    glaciers::Vector{G},
    parameters::Sleipnir.Parameters
    ) where {G <: Sleipnir.AbstractGlacier}

    # Build the results struct based on input values
    inversion = Inversion(model,
                            glaciers,
                            parameters,
                            Vector{InversionMetrics}([]))

    return inversion
end





###############################################
################### UTILS #####################
###############################################

include("inversion_utils.jl")


