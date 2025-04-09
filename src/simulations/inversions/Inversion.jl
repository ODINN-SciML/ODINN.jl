export Inversion

"""
    mutable struct InversionResults{F <: Real}

A structure to store the results of an inversion simulation.

# Fields
- `rgi_id::Union{String, Nothing}`: The RGI identifier, which can be a string or nothing.
- `A::F`: Area parameter.
- `n::F`: Exponent parameter.
- `C::Matrix{F}`: Coefficient matrix.
- `H_pred::Matrix{F}`: Predicted height matrix.
- `H_obs::Matrix{F}`: Observed height matrix.
- `H_diff::Matrix{F}`: Difference between predicted and observed height matrices.
- `V_pred::Matrix{F}`: Predicted volume matrix.
- `V_obs::Matrix{F}`: Observed volume matrix.
- `V_diff::Matrix{F}`: Difference between predicted and observed volume matrices.
- `MSE::F`: Mean squared error.
- `Δx::F`: Grid spacing in the x-direction.
- `Δy::F`: Grid spacing in the y-direction.
- `lon::F`: Longitude.
- `lat::F`: Latitude.
"""
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


"""
    Inversion <: Simulation

A mutable struct that represents an inversion simulation.

# Fields
- `model::Sleipnir.Model`: The model used for the inversion.
- `glaciers::Vector{Sleipnir.AbstractGlacier}`: A vector of glaciers involved in the inversion.
- `parameters::Sleipnir.Parameters`: The parameters used for the inversion.
- `inversion::Vector{InversionResults}`: A vector of results from the inversion.

"""
mutable struct Inversion  <: Simulation 
    model::Sleipnir.Model
    glaciers::Vector{Sleipnir.AbstractGlacier}
    parameters::Sleipnir.Parameters
    inversion::Vector{InversionResults}
end

"""
    Inversion(model::Sleipnir.Model, glaciers::Vector{G}, parameters::Sleipnir.Parameters) where {G <: Sleipnir.AbstractGlacier}

Create an `Inversion` object using the provided model, glaciers, and parameters.

# Arguments
- `model::Sleipnir.Model`: The model to be used for the inversion.
- `glaciers::Vector{G}`: A vector of glaciers, where each glacier is a subtype of `Sleipnir.AbstractGlacier`.
- `parameters::Sleipnir.Parameters`: The parameters to be used for the inversion.

# Returns
- `inversion`: An `Inversion` object initialized with the provided model, glaciers, and parameters.
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


