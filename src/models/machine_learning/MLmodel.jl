
# Abstract type as a parent type for Machine Learning models
abstract type MLmodel end

@kwdef struct NN{F <: AbstractFloat} <: MLmodel 
    architecture::Flux.Chain
    θ::Vector{F}
end

###############################################
################### UTILS #####################
###############################################

include("ML_utils.jl")