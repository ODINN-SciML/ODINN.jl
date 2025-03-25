export AbstractAdjointMethod
export ODINNContinuousAdjoint, ODINNDiscreteAdjoint
export ODINNDummyAdjoint
export ODINNEnzymeAdjoint, ODINNZygoteAdjoint

"""
    AbstractAdjointMethod

Abstract type representing the flavor of AD and adjoint to be used to compute the
gradient of the cost function. There are two parts where one can play with how the
gradient is propagated: the iceflow model VJP and the adjoint of the ODE solver.
The VJP of the iceflow model can be computed using either AD (Zygote or Enzyme), the
discrete, or the continuous adjoint of the iceflow model.
As for the computation of the adjoint of the ODE solution, it can be handled by
SciMLSensitivity, or computed using the adjoint implemented in ODINN.
"""
abstract type AbstractAdjointMethod end

"""
Continuous adjoint of SIA2D with manual implementation of the backward in the ODE
scheme.

`ODINNContinuousAdjoint{F <: AbstractFloat}`

# Fields
- `reltol::F`: Relative tolerance to be used in the ODE solver of the adjoint.
- `abstol::F`: Absolute tolerance to be used in the ODE solver of the adjoint.
"""
@kwdef struct ODINNContinuousAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    reltol::F = 1e-6
    abstol::F = 1e-6
end

"""
Discrete adjoint of SIA2D with manual implementation of the backward in the ODE
scheme.

`ODINNDiscreteAdjoint{F <: AbstractFloat}`

# Fields
- `step::F`: Step size to use in the backward of the ODE.
"""
@kwdef struct ODINNDiscreteAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    step::F = 1/12
end

"""
Struct to provide a dummy gradient. It does not have to be the true gradient.
Mainly used to test that the optimization pileline works independenly of the
gradient calculation.

`ODINNDummyAdjoint`

# Fields:
- `grad::Function`: In-place function `f(du, u; kwargs)` that fills the first
    argument `du` with the gradient values.
"""
@kwdef struct ODINNDummyAdjoint <: AbstractAdjointMethod
    grad::Function
end

"""
Enzyme AD of SIA2D with manual implementation of the backward in the ODE scheme.

`ODINNEnzymeAdjoint{F <: AbstractFloat}`

# Fields
- `step::F`: Step size to use in the backward of the ODE.
"""
@kwdef struct ODINNEnzymeAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    step::F = 1/12
end

"""
Zygote AD of SIA2D with manual implementation of the backward in the ODE scheme.

`ODINNZygoteAdjoint{F <: AbstractFloat}`

# Fields
- `step::F`: Step size to use in the backward of the ODE.
"""
@kwdef struct ODINNZygoteAdjoint{F <: AbstractFloat} <: AbstractAdjointMethod
    step::F = 1/12
end
