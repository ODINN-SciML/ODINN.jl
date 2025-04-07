export AbstractVJPMethod
export DiscreteVJP, ContinuousVJP, EnzymeVJP

"""
    AbstractVJPMethod

Abstract type representing the flavor of AD to be used to compute the
VJP inside the gradient of the cost function.
"""
abstract type AbstractVJPMethod end

"""
Discrete manual implementation of the VJP required inside the adjoint calculation.
This implements the pullback function for the function to differentiate.

`DiscreteVJP`
"""
@kwdef struct DiscreteVJP <: AbstractVJPMethod
end

"""
Continuous manual implementation of the VJP required inside the adjoint calculation.
It relies in the continuous expresion for the adjoint operation based on the functional
for of the forward PDE.

`ContinuousVJP`
"""
@kwdef struct ContinuousVJP <: AbstractVJPMethod
end

"""
Enzyme implementation of VJP used inside the adjoint calculation.

`EnzymeVJP`
"""
@kwdef struct EnzymeVJP <: AbstractVJPMethod
end