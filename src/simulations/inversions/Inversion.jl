export Inversion

mutable struct InversionResults{F <: Real} 
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

Base.:(==)(a::InversionResults, b::InversionResults) = 
    a.rgi_id == b.rgi_id &&
    a.A == b.A &&
    a.n == b.n &&
    a.C .== b.C &&
    all(a.C .== b.C) &&
    all(a.H_pred .== b.H_pred) &&
    all(a.H_obs .== b.H_obs) &&
    all(a.H_diff .== b.H_diff) &&
    all(a.V_pred .== b.V_pred) &&
    all(a.V_obs .== b.V_obs) &&
    all(a.V_diff .== b.V_diff) &&
    a.MSE == b.MSE &&
    a.Δx == b.Δx &&
    a.Δy == b.Δy &&
    a.lon == b.lon &&
    a.lat == b.lat


mutable struct Inversion  <: Simulation 
    model::Sleipnir.Model
    glaciers::Vector{Sleipnir.AbstractGlacier}
    parameters::Sleipnir.Parameters
    inversion::Vector{InversionResults}
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
                            Vector{InversionResults}([]))

    return inversion
end





###############################################
################### UTILS #####################
###############################################

include("inversion_utils.jl")


