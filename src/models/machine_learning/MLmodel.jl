export NeuralNetwork, Model

_inputs_A_law = (; T=iTemp())
_inputs_C_law = (; )
_inputs_n_law = (; )
_inputs_Y_law = (; T=iTemp(), H̄=iH̄())
_inputs_U_law = (; H̄=iH̄(), ∇S=i∇S())

"""
    OptimizableModel <: AbstractModel

Optimizable components of the model.
"""

abstract type OptimizableModel <: AbstractModel end
"""
    PerGlacierModel <: OptimizableModel

Abstract type representing per glacier optimizable components of the model.
This is a subtype of `OptimizableModel`.
Typically used for classical inversions.
"""
abstract type PerGlacierModel <: OptimizableModel end

"""
    ParameterizationModel <: OptimizableModel

Abstract type representing parameterizable components of the model.
This is a subtype of `OptimizableModel`.
Typically used for functional inversions.
"""
abstract type ParameterizationModel <: OptimizableModel end

include("./InitialCondition.jl")
include("./GlacierWideInv.jl")

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
        if haskey(regressors, :U) && regressors.U isa ParameterizationModel
            @assert inputs(iceflow.U)==_inputs_U_law "Inputs of U law must be $(_inputs_U_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.U))."
        end
    elseif iceflow.Y_is_provided
        if haskey(regressors, :Y) && regressors.Y isa ParameterizationModel
            @assert inputs(iceflow.Y)==_inputs_Y_law "Inputs of Y law must be $(_inputs_Y_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.Y))."
        end
    else
        if haskey(regressors, :A) && regressors.A isa ParameterizationModel
            @assert inputs(iceflow.A)==_inputs_A_law "Inputs of A law must be $(_inputs_A_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.A))."
        end
        if haskey(regressors, :C) && regressors.C isa ParameterizationModel
            @assert inputs(iceflow.C)==_inputs_C_law "Inputs of C law must be $(_inputs_C_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.C))."
        end
        if haskey(regressors, :n) && regressors.n isa ParameterizationModel
            @assert inputs(iceflow.n)==_inputs_n_law "Inputs of n law must be $(_inputs_n_law) in ODINN for functional inversions but the ones provided are $(inputs(iceflow.n))."
        end
    end

    # Build target based on parameters
    if isnothing(target)
        if iceflow.U_is_provided
            target = SIA2D_D_target()
        elseif iceflow.Y_is_provided
            target = SIA2D_D_hybrid_target()
        elseif !(haskey(regressors, :A) && regressors.A isa ParameterizationModel) || inputs(iceflow.A)==_inputs_A_law
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

    machine_learning = MachineLearning(target, regressors)
    return Sleipnir.Model(iceflow, mass_balance, machine_learning)
end

"""
    NeuralNetwork{
        ChainType <: Lux.Chain,
        ComponentVectorType <: ComponentVector,
        NamedTupleType <: NamedTuple,
    } <: ParameterizationModel

Feed-forward neural network.

# Fields
- `architecture::ChainType`: `Flux.Chain` neural network architecture
- `θ::ComponentVectorType`: Neural network parameters
- `st::NamedTupleType`: Neural network status
"""
mutable struct NeuralNetwork{
    ChainType <: Lux.Chain,
    ComponentVectorType <: ComponentVector,
    NamedTupleType <: NamedTuple,
} <: ParameterizationModel
    architecture::ChainType
    θ::ComponentVectorType
    st::NamedTupleType

    function NeuralNetwork(
        params::P;
        architecture::Union{ChainType, Nothing} = nothing,
        θ::Union{ComponentArrayType, Nothing} = nothing,
        st::Union{NamedTupleType, Nothing} = nothing,
        seed::Union{RNG, Nothing} = nothing,
    ) where {
        P<:Sleipnir.Parameters,
        ChainType<:Lux.Chain,
        ComponentArrayType<:ComponentArray,
        NamedTupleType<:NamedTuple,
        RNG<:AbstractRNG
    }

        # Float type
        ft = Sleipnir.Float
        lightNN = params.simulation.test_mode

        if isnothing(architecture) & isnothing(θ) & isnothing(st)
            if params.UDE.target == :A
                architecture, θ, st = get_default_NN(θ, ft; lightNN = lightNN, seed = seed)
            elseif params.UDE.target == :D_hybrid
                architecture = build_default_NN(; n_input = 2, lightNN = lightNN)
                architecture, θ, st = set_NN(architecture; ft = ft, seed = seed)
            elseif params.UDE.target == :D
                architecture = build_default_NN(; n_input = 2, lightNN = lightNN)
                architecture, θ, st = set_NN(architecture; ft = ft, seed = seed)
            else
                @warn "Constructing default Neural Network"
                architecture, θ, st = get_default_NN(θ, ft; lightNN = lightNN, seed = seed)
            end
        elseif !isnothing(architecture) & isnothing(θ) & isnothing(st)
            architecture, θ, st = set_NN(architecture; ft = ft, seed = seed)
        elseif !isnothing(architecture) & !isnothing(θ) & isnothing(st)
            architecture, θ, st = set_NN(architecture; θ_trained = θ, ft = ft, seed = seed)
        elseif !isnothing(architecture) & !isnothing(θ) & !isnothing(st)
            # Architecture and setup already provided
            nothing
        else
            @warn "To specify the neural network please provide architecture, (architecture, θ, st), or none of them to create default NN."
        end

        new{typeof(architecture), typeof(θ), typeof(st)}(
            architecture, θ, st
        )
    end

end
# Note: we could define any other kind of regressor as a subtype of OptimizableModel

# Empty optimizable model
struct emptyOptimizableModel <: OptimizableModel end

mutable struct MachineLearning{
    MLmodelAType <: OptimizableModel,
    MLmodelCType <: OptimizableModel,
    MLmodelnType <: OptimizableModel,
    MLmodelYType <: OptimizableModel,
    MLmodelUType <: OptimizableModel,
    MLmodelICType <: OptimizableModel,
    TAR <: AbstractTarget,
    ComponentArrayType <: ComponentArray,
} <: AbstractModel
    A::Union{MLmodelAType, Nothing}
    C::Union{MLmodelCType, Nothing}
    n::Union{MLmodelnType, Nothing}
    Y::Union{MLmodelYType, Nothing}
    U::Union{MLmodelUType, Nothing}
    IC::Union{MLmodelICType, Nothing}
    target::TAR
    θ::Union{ComponentArrayType, Nothing}

    function MachineLearning(
        target,
        regressors::NamedTuple = (;)
    )
        θ = ComponentVector(; (k => r.θ.θ for (k, r) in pairs(regressors))...)
        if length(θ) == 0
            θ = nothing
        end
        A = haskey(regressors, :A) ? regressors.A : emptyOptimizableModel()
        C = haskey(regressors, :C) ? regressors.C : emptyOptimizableModel()
        n = haskey(regressors, :n) ? regressors.n : emptyOptimizableModel()
        Y = haskey(regressors, :Y) ? regressors.Y : emptyOptimizableModel()
        U = haskey(regressors, :U) ? regressors.U : emptyOptimizableModel()
        # Dedicated regressor for initial condition
        IC = haskey(regressors, :IC) ? regressors.IC : emptyIC()

        new{typeof(A), typeof(C), typeof(n), typeof(Y), typeof(U), typeof(IC), typeof(target), typeof(θ)}(A, C, n, Y, U, IC, target, θ)
    end
    function MachineLearning(
        ml::MachineLearning,
        θ::Union{ComponentArray, Nothing},
    )
        new{
            typeof(ml.A), typeof(ml.C), typeof(ml.n), typeof(ml.Y), typeof(ml.U),
            typeof(ml.IC), typeof(ml.target), typeof(θ)
        }(ml.A, ml.C, ml.n, ml.Y, ml.U, ml.IC, ml.target, θ)
    end
end

"""
    splitθ(θ, glacier_idx::Integer, optimizableComponent::OptimizableModel)

Given a `ComponentVector` `θ`, a `glacier_idx` and an `optimizableComponent`, extract the content
of `θ` relevant for the given `optimizableComponent` and glacier ID `glacier_idx`.
"""
function splitθ(θ, glacier_idx::Integer, optimizableComponent::OptimizableModel)
    if isa(optimizableComponent, PerGlacierModel)
        glacier_id = Symbol("$(glacier_idx)")
        return ComponentVector(NamedTuple{(Symbol("1"),)}((θ[glacier_id],)))
    else
        return θ
    end
end
function splitθ(θ::ComponentArray, glacier_idx::Integer, ml::MachineLearning)
    return ComponentVector(; map(k -> (k=>splitθ(θ[k], glacier_idx, getfield(ml, k))), keys(θ))...)
end
"""
    aggregate∇θ(∇θ::Vector{<: ComponentArray}, θ, ml::MachineLearning)

Aggregate the vector of gradients `∇θ` as a single `ComponentArray`.
The argument `∇θ` is the vector of all the gradients computed for each glacier.
This function aggregates them based on the optimizable components of `ml`.
"""
function aggregate∇θ(∇θ::Vector{<: ComponentArray}, θ, ml::MachineLearning)
    ∇θfull = Dict()
    for k in keys(θ)
        optimizableComponent = getfield(ml, k)
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

# Display setup
function Base.show(io::IO, nn_model::NeuralNetwork)
    println(io, "--- NeuralNetwork ---")
    println(io, "    architecture:")
    # Retrieve the printed lines
    iotmp = IOBuffer()
    show(iotmp, "text/plain", nn_model.architecture)
    str = String(take!(iotmp))
    # Add prefix to each line
    prefix = "      "
    prefixed_str = join(prefix .* split(str, '\n'), '\n')
    println(io, prefixed_str)
    print(io, "    θ: ComponentVector of length $(length(nn_model.θ))")
end

function Base.show(io::IO, ml_model::MachineLearning)
    if !(ml_model.A isa emptyOptimizableModel)
        print(io, "  A: ")
        println(io, ml_model.A)
    end
    if !(ml_model.C isa emptyOptimizableModel)
        print(io, "  C: ")
        println(io, ml_model.C)
    end
    if !(ml_model.n isa emptyOptimizableModel)
        print(io, "  n: ")
        println(io, ml_model.n)
    end
    if !(ml_model.Y isa emptyOptimizableModel)
        print(io, "  Y: ")
        println(io, ml_model.Y)
    end
    if !(ml_model.U isa emptyOptimizableModel)
        print(io, "  U: ")
        println(io, ml_model.U)
    end
    if !(ml_model.IC isa emptyIC)
        print(io, "  IC: ")
        println(io, ml_model.IC)
    end
end
