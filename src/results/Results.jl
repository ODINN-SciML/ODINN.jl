export Result
export save_simulation_file!

abstract type AbstractResult end

@kwdef struct Result{F <: AbstractFloat} <: AbstractResult
    θ::ComponentVector
    losses::Vector{F}
    params::Sleipnir.Parameters
end

include("result_utils.jl")