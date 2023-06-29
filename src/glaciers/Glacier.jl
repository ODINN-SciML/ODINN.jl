
include("Climate.jl")

@kwdef mutable struct Glacier{F <: AbstractFloat, I <: Int} 
    rgi_id::String
    gdir::PyObject 
    climate::Union{Climate, Nothing}
    H₀::Matrix{F}
    S::Matrix{F}
    B::Matrix{F}
    V::Matrix{F}
    slope::Matrix{F}
    dist_border::Matrix{F}
    S_coords::PyObject
    Δx::F
    Δy::F
    nx::I
    ny::I
end

###############################################
################### UTILS #####################
###############################################

include("climate_utils.jl")
include("glacier_utils.jl")

