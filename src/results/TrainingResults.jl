export Result
export save_inversion_file!

abstract type AbstractResult end

@kwdef struct TrainingResult{F <: AbstractFloat} <: AbstractResult
    θ::ComponentVector
    θ_hist::Vector{ComponentVector}
    ∇θ_hist::Vector{ComponentVector}
    losses::Vector{F}
    params::Sleipnir.Parameters
end

include("trainingresult_utils.jl")
