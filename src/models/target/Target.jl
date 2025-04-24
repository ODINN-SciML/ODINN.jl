export AbstractTarget, AbstractSIA2DTarget
export SIA2D_target
export ComponentVector2Vector, Vector2ComponentVector
export predict_A̅

abstract type AbstractTarget end
abstract type AbstractSIA2DTarget <: AbstractTarget end

"""
    SIA2D_target <: AbstractSIA2DTarget

Implementation of Target objective for learning elements of the SIA equation.

/!\\ Since this struct declares several functions, which have an abstract type, it is important
to create constructor functions which explicitly declare the type of the function
that is passed to the constructor. Otherwise, the compiler will not be able to infer
the type of the function and will not be able to compile the code. 

# Fields
- `name::Symbol`: A symbolic identifier for the target model.
- `D::Function`: A function `D(H, ∇H, θ)` defining the SIA diffusivity.
- `∂D∂H::Function`: Partial derivative of the diffusivity `D` with respect to ice thickness `H`.
- `∂D∂∇H::Function`: Partial derivative of `D` with respect to the gradient of ice thickness `∇H`.
- `∂D∂θ::Function`: Partial derivative of `D` with respect to the parameter field `θ`.
- `apply_parametrization::Function`: A function to apply a parametrization to the model.
- `apply_parametrization!::Function`: An in-place version of `apply_parametrization`.
"""
struct SIA2D_target{FD, FDH, FDHH, FDθ, FP, FP!} <: AbstractSIA2DTarget
    name::Symbol
    D::FD
    ∂D∂H::FDH
    ∂D∂∇H::FDHH
    ∂D∂θ::FDθ
    apply_parametrization::FP
    apply_parametrization!::FP!
end

"""
    function SIA2D_target(;
       name::Symbol = :A,
    )

Constructor of the SIA target. All the relevant functions defined inside Target are
constructed automatically by just providing the keyword `name` for the inversion.

# Arguments
- `name::Symbol`: Identifying name for the model inversion.
"""
function SIA2D_target(;
    name::Symbol = :A,
    interpolation::Bool = true
)
    if name == :foo
        build_target_foo()
    elseif name == :A
        build_target_A()
    elseif name == :D
        build_target_D(; interpolation = interpolation)
    else
        @error "Target method named $(name) not implemented."
    end
end

### Add specific target objectives
include("target_utils.jl")
include("target_A.jl")
include("target_D.jl")