
include("Climate.jl")

@kwdef struct Glacier{F <: AbstractFloat} 
    rgi_id::String
    gdir::PyObject 
    climate::Climate
    H₀::Matrix{F}
    S::Matrix{F}
    B::Matrix{F}
    V::Matrix{F}
    slope::Matrix{F}
    dist_border::Matrix{F}
    S_coords::PyObject
    Δx::F
    Δy::F
    nx::F
    ny::F
end

###############################################
################### UTILS #####################
###############################################

include("glacier_utils.jl")
include("climate_utils.jl")

