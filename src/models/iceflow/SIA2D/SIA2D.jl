
include("../IceflowModel.jl")

###############################################
###### SHALLOW ICE APPROXIMATION MODELS #######
###############################################

@kwdef mutable struct SIA2Dmodel{F <: AbstractFloat} <: SIAmodel
    A::F
    H::Matrix{F}
    S::Matrix{F}
    dSdx::Matrix{F}
    dSdy::Matrix{F}
    D::Matrix{F}
    dSdx_edges::Matrix{F}
    dSdy_edges::Matrix{F}
    ∇S::Matrix{F}
    Fx::Matrix{F}
    Fy::Matrix{F}
    V::Matrix{F}
    Γ::F
end

###############################################
################### UTILS #####################
###############################################

include("SIA2D_utils.jl")


