export Inversion

mutable struct InversionParams{F <: AbstractFloat}
    A::F
    n::F
    C::F
end

function InversionParams(A::F, n::F, C::F) where {F <: AbstractFloat}
    return InversionParams{F}(A, n, C)
end

mutable struct Inversion  <: Simulation 
    model::Sleipnir.Model
    glaciers::Vector{Sleipnir.AbstractGlacier}
    parameters::Sleipnir.Parameters
    inversion::Vector{InversionParams}
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
                            Vector{InversionParams}([]))

    return inversion
end





###############################################
################### UTILS #####################
###############################################

include("inversion_utils.jl")


