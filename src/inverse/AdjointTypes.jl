export AbstractAdjointMethod
export ODINNContinuousAdjoint, ODINNDiscreteAdjoint
export ODINNDummyAdjoint
export ODINNEnzymeAdjoint, ODINNZygoteAdjoint

abstract type AbstractAdjointMethod end

@kwdef struct ODINNContinuousAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    # Continuous adjoint of SIA2D with manual implementation of the backward in the ODE scheme
    reltol::F = 1e-6
    abstol::F = 1e-6
end

@kwdef struct ODINNDiscreteAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    # Discrete adjoint of SIA2D with manual implementation of the backward in the ODE scheme
    step::F = 1/12
end

@kwdef struct ODINNDummyAdjoint <: AbstractAdjointMethod
    grad::Function
end

@kwdef struct ODINNEnzymeAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    # Enzyme AD of SIA2D with manual implementation of the backward in the ODE scheme
    step::F = 1/12
end

@kwdef struct ODINNZygoteAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    # Zygote AD of SIA2D with manual implementation of the backward in the ODE scheme
    step::F = 1/12
end
