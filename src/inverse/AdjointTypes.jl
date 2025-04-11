export AbstractAdjointMethod
export ContinuousAdjoint, DiscreteAdjoint
export DummyAdjoint

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
Struct that defines the SciMLSensitivity adjoint flavor. This is the default
behavior in ODINN.

`SciMLSensitivityAdjoint`
"""
@kwdef struct SciMLSensitivityAdjoint <: AbstractAdjointMethod
end

"""
    ContinuousAdjoint{
        F <: AbstractFloat,
        I <: Integer,
        VJP <: AbstractVJPMethod
        } <: AbstractAdjointMethod

Continuous adjoint of SIA2D with manual implementation of the backward in the ODE
scheme.

# Fields
- `VJP_method::VJP`: Type of AbstractVJPMethod used to compute VJPs inside adjoint
    calculation.
- `solver::Any`: The solver to be used for adjoint.
- `reltol::F`: Relative tolerance to be used in the ODE solver of the adjoint.
- `abstol::F`: Absolute tolerance to be used in the ODE solver of the adjoint.
- `n_quadrature::I`: Number of nodes used in the Gauss quadrature for the numerical
    integration of the loss function.
"""
@kwdef struct ContinuousAdjoint{
    F <: AbstractFloat,
    I <: Integer,
    VJP <: AbstractVJPMethod
} <: AbstractAdjointMethod
    VJP_method::VJP = DiscreteVJP()
    solver::Any = RDPK3Sp35()
    reltol::F = 1e-8
    abstol::F = 1e-8
    dtmax::F = 1/12
    n_quadrature::I = 200
end

"""
    DiscreteAdjoint{VJP <: AbstractVJPMethod} <: AbstractAdjointMethod

Discrete adjoint of SIA2D with manual implementation of the backward in the ODE
scheme.

# Fields
- `VJP_method`: Type of AbstractVJPMethod used to compute VJPs inside adjoint
    calculation.
"""
@kwdef struct DiscreteAdjoint{VJP <: AbstractVJPMethod} <: AbstractAdjointMethod
    VJP_method::VJP = DiscreteVJP()
end

"""
Struct to provide a dummy gradient. It does not have to be the true gradient.
Mainly used to test that the optimization pileline works independenly of the
gradient calculation.

`DummyAdjoint`

# Fields:
- `grad::Function`: In-place function `f(du, u; kwargs)` that fills the first
    argument `du` with the gradient values.
"""
@kwdef struct DummyAdjoint <: AbstractAdjointMethod
    grad_function::Union{Function, Nothing} = nothing
end
