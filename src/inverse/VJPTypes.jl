export AbstractVJPMethod
export DiscreteVJP, ContinuousVJP, EnzymeVJP, NoVJP

"""
    AbstractVJPMethod

Abstract type representing the flavor of AD to be used to compute the
VJP inside the gradient of the cost function.
"""
abstract type AbstractVJPMethod end

"""
    DiscreteVJP{ADTYPE <: DI.AbstractADType} <: AbstractVJPMethod

Discrete manual implementation of the VJP required inside the adjoint calculation.
This implements the pullback function for the function to differentiate.

# Fields
- `regressorADBackend::ADTYPE`: Specifies the AD backend used for the laws when
    their associated VJPs functions are not provided. The type parameter `ADTYPE`
    must be a subtype of `DI.AbstractADType`.

# Constructor
- The default constructor allows specifying the backend via the `regressorADBackend`
    keyword argument, defaulting to `DI.AutoMooncake()`.
"""
struct DiscreteVJP{ADTYPE <: DI.AbstractADType} <: AbstractVJPMethod
    regressorADBackend::ADTYPE

    function DiscreteVJP(;
        regressorADBackend = DI.AutoMooncake(),
    )
        new{typeof(regressorADBackend)}(regressorADBackend)
    end
end

"""
    ContinuousVJP{ADTYPE <: DI.AbstractADType} <: AbstractVJPMethod

Continuous manual implementation of the VJP required inside the adjoint calculation.
It relies in the continuous expresion for the adjoint operation based on the functional
formula of the forward PDE.

# Fields
- `regressorADBackend::ADTYPE`: Specifies the AD backend used for the laws when
    their associated VJPs functions are not provided. The type parameter `ADTYPE`
    must be a subtype of `DI.AbstractADType`.

# Constructor
- The default constructor allows specifying the backend via the `regressorADBackend`
    keyword argument, defaulting to `DI.AutoMooncake()`.
"""
struct ContinuousVJP{ADTYPE <: DI.AbstractADType} <: AbstractVJPMethod
    regressorADBackend::ADTYPE

    function ContinuousVJP(;
        regressorADBackend = DI.AutoMooncake(),
    )
        new{typeof(regressorADBackend)}(regressorADBackend)
    end
end

"""
Enzyme implementation of VJP used inside the adjoint calculation.

`EnzymeVJP`
"""
struct EnzymeVJP <: AbstractVJPMethod
end

"""
No VJP flavor when the contribution of a given term should not be computed inside the adjoint calculation (e.g. MB).

`NoVJP`
"""
struct NoVJP <: AbstractVJPMethod
end
