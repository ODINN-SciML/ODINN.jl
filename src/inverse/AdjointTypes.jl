export AbstractAdjointMethod
export ODINNContinuousAdjoint, ODINNDiscreteAdjoint
export ODINNDummyAdjoint
export ODINNEnzymeAdjoint, ODINNZygoteAdjoint

abstract type AbstractAdjointMethod end

@kwdef struct ODINNContinuousAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    reltol::F = 1e-6
    abstol::F = 1e-6
end

@kwdef struct ODINNDiscreteAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    step::F = 1/12
end

@kwdef struct ODINNDummyAdjoint <: AbstractAdjointMethod
    grad::Function
end

@kwdef struct ODINNEnzymeAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    step::F = 1/12
end

@kwdef struct ODINNZygoteAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    step::F = 1/12
end