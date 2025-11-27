export Model

_inputs_A_law_scalar = (; T=iAvgScalarTemp())
_inputs_A_law_gridded = (; T=iAvgGriddedTemp())
_inputs_C_law = (; )
_inputs_n_law = (; )
_inputs_Y_law = (; T=iAvgScalarTemp(), H̄=iH̄())
_inputs_U_law = (; H̄=iH̄(), ∇S=i∇S())

"""
    TrainableModel <: AbstractModel

Trainable components of the model.
This can be either functional models or classical inversion models.
"""

abstract type TrainableModel <: AbstractModel end
"""
    PerGlacierModel <: TrainableModel

Abstract type representing per glacier optimizable components of the model.
This is a subtype of `TrainableModel`.
Typically used for classical inversions.
"""
abstract type PerGlacierModel <: TrainableModel end

"""
    FunctionalModel <: TrainableModel

Abstract type representing functional learnable components of the model.
This is a subtype of `TrainableModel`.
Typically used for functional inversions.
"""
abstract type FunctionalModel <: TrainableModel end

include("./NeuralNetwork.jl")
include("./InitialCondition.jl")
include("./GlacierWideInv.jl")
include("./GriddedInv.jl")

"""
    Model(;
        iceflow::Union{IFM, Nothing} = nothing,
        mass_balance::Union{MBM, Nothing} = nothing,
        regressors::Union{NamedTuple, Nothing} = nothing,
        target::Union{TAR, Nothing} = nothing,
    ) where {IFM <: IceflowModel, MBM <: MBmodel, TAR <: AbstractTarget}

Creates a new model instance using the provided iceflow, mass balance, and machine learning components.

# Arguments
- `iceflow::Union{IFM, Nothing}`: The iceflow model to be used. Can be a single model or `nothing`.
- `mass_balance::Union{MBM, Nothing}`: The mass balance model to be used. Can be a single model or `nothing`.
- `regressors::Union{NamedTuple, Nothing}`: The regressors to be used in the laws.
# Returns
- `model`: A new instance of `Sleipnir.Model` initialized with the provided components.
"""
function Model(;
    iceflow::Union{IFM, Nothing} = nothing,
    mass_balance::Union{MBM, Nothing} = nothing,
    regressors::Union{NamedTuple, Nothing} = nothing,
    target::Union{TAR, Nothing} = nothing,
) where {IFM <: IceflowModel, MBM <: MBmodel, TAR <: AbstractTarget}
    if isnothing(regressors)
        Sleipnir.Model(iceflow, mass_balance, nothing)
    else
        Model(iceflow, mass_balance, regressors; target=target)
    end
end
function Model(
    iceflow::Union{IFM, Nothing},
    mass_balance::Union{MBM, Nothing},
    regressors::NamedTuple;
    target::Union{TAR, Nothing} = nothing,
) where {IFM <: IceflowModel, MBM <: MBmodel, TAR <: AbstractTarget}

    # Check that the inputs match what is hardcoded in the adjoint computation when the regressor is used in the context of a functional inversion
    if iceflow.U_is_provided
        if haskey(regressors, :U) && regressors.U isa FunctionalModel
            @assert inputs(iceflow.U)==_inputs_U_law "Inputs of U law must be $(_inputs_U_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.U))."
        end
    elseif iceflow.Y_is_provided
        if haskey(regressors, :Y) && regressors.Y isa FunctionalModel
            @assert inputs(iceflow.Y)==_inputs_Y_law "Inputs of Y law must be $(_inputs_Y_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.Y))."
        end
    else
        if haskey(regressors, :A) && regressors.A isa FunctionalModel
            @assert inputs(iceflow.A)==_inputs_A_law_scalar || inputs(iceflow.A)==_inputs_A_law_gridded "Inputs of A law must be $(_inputs_A_law_scalar) or $(_inputs_A_law_gridded) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.A))."
        end
        if haskey(regressors, :C) && regressors.C isa FunctionalModel
            @assert inputs(iceflow.C)==_inputs_C_law "Inputs of C law must be $(_inputs_C_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.C))."
        end
        if haskey(regressors, :n) && regressors.n isa FunctionalModel
            @assert inputs(iceflow.n)==_inputs_n_law "Inputs of n law must be $(_inputs_n_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.n))."
        end
    end

    # Build target based on parameters
    if isnothing(target)
        if iceflow.U_is_provided
            target = SIA2D_D_target()
        elseif iceflow.Y_is_provided
            target = SIA2D_D_hybrid_target()
        elseif !(haskey(regressors, :A) && regressors.A isa FunctionalModel) || inputs(iceflow.A)==_inputs_A_law_scalar || inputs(iceflow.A)==_inputs_A_law_gridded
            target = SIA2D_A_target()
        else
            throw("Cannot infer target from the laws.")
        end
    else
        if iceflow.U_is_provided
            @assert targetType(target) == :D "The provided laws do not match with the provided target. Make sure that the target is a SIA2D_D_target."
        elseif iceflow.Y_is_provided
            @assert targetType(target) == :D_hybrid "The provided laws do not match with the provided target. Make sure that the target is a SIA2D_D_hybrid_target."
        else
            @assert targetType(target) == :A "The provided laws do not match with the provided target. Make sure that the target is a SIA2D_A_target."
        end
    end

    trainable_components = TrainableComponents(target, regressors)
    return Sleipnir.Model(iceflow, mass_balance, trainable_components)
end

# Empty optimizable model
struct emptyTrainableModel <: TrainableModel end

mutable struct TrainableComponents{
    TrainableModelAType <: TrainableModel,
    TrainableModelCType <: TrainableModel,
    TrainableModelnType <: TrainableModel,
    TrainableModelYType <: TrainableModel,
    TrainableModelUType <: TrainableModel,
    TrainableModelICType <: TrainableModel,
    TAR <: AbstractTarget,
    ComponentArrayType <: ComponentArray,
} <: AbstractModel
    A::Union{TrainableModelAType, Nothing}
    C::Union{TrainableModelCType, Nothing}
    n::Union{TrainableModelnType, Nothing}
    Y::Union{TrainableModelYType, Nothing}
    U::Union{TrainableModelUType, Nothing}
    IC::Union{TrainableModelICType, Nothing}
    target::TAR
    θ::Union{ComponentArrayType, Nothing}

    function TrainableComponents(
        target,
        regressors::NamedTuple = (;)
    )
        θ = ComponentVector(; (k => r.θ.θ for (k, r) in pairs(regressors))...)
        if length(θ) == 0
            θ = nothing
        end
        A = haskey(regressors, :A) ? regressors.A : emptyTrainableModel()
        C = haskey(regressors, :C) ? regressors.C : emptyTrainableModel()
        n = haskey(regressors, :n) ? regressors.n : emptyTrainableModel()
        Y = haskey(regressors, :Y) ? regressors.Y : emptyTrainableModel()
        U = haskey(regressors, :U) ? regressors.U : emptyTrainableModel()
        # Dedicated regressor for initial condition
        IC = haskey(regressors, :IC) ? regressors.IC : emptyIC()

        new{typeof(A), typeof(C), typeof(n), typeof(Y), typeof(U), typeof(IC), typeof(target), typeof(θ)}(A, C, n, Y, U, IC, target, θ)
    end
    function TrainableComponents(
        submodels::TrainableComponents,
        θ::Union{ComponentArray, Nothing},
    )
        new{
            typeof(submodels.A), typeof(submodels.C), typeof(submodels.n), typeof(submodels.Y), typeof(submodels.U),
            typeof(submodels.IC), typeof(submodels.target), typeof(θ)
        }(submodels.A, submodels.C, submodels.n, submodels.Y, submodels.U, submodels.IC, submodels.target, θ)
    end
end

"""
    splitθ(θ, glacier_idx::Integer, optimizableComponent::TrainableModel)

Given a `ComponentVector` `θ`, a `glacier_idx` and an `optimizableComponent`, extract the content
of `θ` relevant for the given `optimizableComponent` and glacier ID `glacier_idx`.
"""
function splitθ(θ, glacier_idx::Integer, optimizableComponent::TrainableModel)
    if isa(optimizableComponent, PerGlacierModel)
        glacier_id = Symbol("$(glacier_idx)")
        return ComponentVector(NamedTuple{(Symbol("1"),)}((θ[glacier_id],)))
    else
        return θ
    end
end
function splitθ(θ::ComponentArray, glacier_idx::Integer, submodels::TrainableComponents)
    return ComponentVector(; map(k -> (k=>splitθ(θ[k], glacier_idx, getfield(submodels, k))), keys(θ))...)
end
"""
    aggregate∇θ(∇θ::Vector{<: ComponentArray}, θ, submodels::TrainableComponents)

Aggregate the vector of gradients `∇θ` as a single `ComponentArray`.
The argument `∇θ` is the vector of all the gradients computed for each glacier.
This function aggregates them based on the optimizable components of `submodels`.
"""
function aggregate∇θ(∇θ::Vector{<: ComponentArray}, θ, submodels::TrainableComponents)
    ∇θfull = Dict()
    for k in keys(θ)
        optimizableComponent = getfield(submodels, k)
        tmp_k = zero(θ[k])
        for i in 1:length(∇θ)
            if isa(optimizableComponent, PerGlacierModel)
                glacier_id = Symbol("$(i)")
                tmp_k[glacier_id] .+= ∇θ[i][k][Symbol("1")]
            else
                tmp_k .+= ∇θ[i][k]
            end
        end
        ∇θfull[k] = tmp_k
    end
    return ComponentVector(; ∇θfull...)
end

function Base.show(io::IO, submodels::TrainableComponents)
    if !(submodels.A isa emptyTrainableModel)
        print(io, "  A: ")
        println(io, submodels.A)
    end
    if !(submodels.C isa emptyTrainableModel)
        print(io, "  C: ")
        println(io, submodels.C)
    end
    if !(submodels.n isa emptyTrainableModel)
        print(io, "  n: ")
        println(io, submodels.n)
    end
    if !(submodels.Y isa emptyTrainableModel)
        print(io, "  Y: ")
        println(io, submodels.Y)
    end
    if !(submodels.U isa emptyTrainableModel)
        print(io, "  U: ")
        println(io, submodels.U)
    end
    if !(submodels.IC isa emptyIC)
        print(io, "  IC: ")
        println(io, submodels.IC)
    end
end
