export NeuralNetwork

_inputs_A_law = (; T=InpTemp())
_inputs_Ahybrid_law = (; T=InpTemp(), H̄=InpH̄())
_inputs_C_law = (; )
_inputs_n_law = (; )
_inputs_U_law = (; H̄=InpH̄(), ∇S=Inp∇S())

# Abstract type as a parent type for Machine Learning models
abstract type MLmodel <: AbstractModel end

"""
    Model(; iceflow::Union{IFM, Nothing}, mass_balance::Union{MBM, Nothing}, machine_learning::Union{MLM, Nothing}) where {IFM <: IceflowModel, MBM <: MBmodel, MLM <: MLmodel}

Creates a new model instance using the provided iceflow, mass balance, and machine learning components.

# Arguments
- `iceflow::Union{IFM, Nothing}`: The iceflow model to be used. Can be a single model or `nothing`.
- `mass_balance::Union{MBM, Nothing}`: The mass balance model to be used. Can be a single model or `nothing`.
- `machine_learning::Union{MLM, Nothing}`: The machine learning model to be used. Can be a single model or `nothing`.
# Returns
- `model`: A new instance of `Sleipnir.Model` initialized with the provided components.
"""
function Model(;
    iceflow::Union{IFM, Nothing},
    mass_balance::Union{MBM, Nothing},
    regressors::NamedTuple = (;),
    target::Union{TAR, Nothing} = nothing,
) where {IFM <: IceflowModel, MBM <: MBmodel, TAR <: AbstractTarget}

    if iceflow.U_is_provided
        @assert inputs(iceflow.U)==_inputs_U_law "Inputs of U law must be $(_inputs_U_law) in ODINN."
    else
        @assert inputs(iceflow.A)==_inputs_A_law || inputs(iceflow.A)==_inputs_Ahybrid_law "Inputs of A law must be $(_inputs_A_law) or $(_inputs_Ahybrid_law) in ODINN."
        @assert inputs(iceflow.C)==_inputs_C_law "Inputs of C law must be $(_inputs_C_law) in ODINN."
        @assert inputs(iceflow.n)==_inputs_n_law "Inputs of n law must be $(_inputs_n_law) in ODINN."
    end

    # Build target based on parameters
    if isnothing(target)
        if iceflow.U_is_provided
            target = SIA2D_D_target()
        elseif inputs(iceflow.A)==_inputs_A_law
            target = SIA2D_A_target()
        elseif inputs(iceflow.A)==_inputs_Ahybrid_law
            target = SIA2D_D_hybrid_target()
        else
            throw("Cannot infer target from the laws.")
        end
    else
        if iceflow.U_is_provided
            @assert targetType(target) == :D "The provided laws do not match with the provided target. Make sure that the target is a SIA2D_D_target."
        else
            @assert targetType(target) == :A "The provided laws do not match with the provided target. Make sure that the target is a SIA2D_A_target."
        end
    end

    machine_learning = MachineLearning(target, regressors)

    iceflowType = typeof(iceflow)
    massbalanceType = typeof(mass_balance)
    model = Sleipnir.Model{iceflowType, massbalanceType, typeof(machine_learning)}(iceflow, mass_balance, machine_learning)

    return model
end

"""
    NeuralNetwork{
        ChainType <: Lux.Chain,
        ComponentVectorType <: ComponentVector,
        NamedTupleType <: NamedTuple,
    } <: MLmodel

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
} <: MLmodel
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
# Note: we could define any other kind of regressor as a subtype of MLmodel


struct emptyMLmodel <: MLmodel end

mutable struct MachineLearning{
    MLmodelAType <: MLmodel,
    MLmodelCType <: MLmodel,
    MLmodelnType <: MLmodel,
    MLmodelUType <: MLmodel,
    TAR <: AbstractTarget,
    ComponentArrayType <: ComponentArray,
} <: AbstractModel
    A::Union{MLmodelAType, Nothing}
    C::Union{MLmodelCType, Nothing}
    n::Union{MLmodelnType, Nothing}
    U::Union{MLmodelUType, Nothing}
    target::TAR
    θ::Union{ComponentArrayType, Nothing}

    function MachineLearning(
        target,
        regressors::NamedTuple = (;)
    )
        θ = ComponentVector(; (k => r.θ.θ for (k,r) in pairs(regressors))...)
        if length(θ)==0
            θ = nothing
        end
        A = haskey(regressors, :A) ? regressors.A : emptyMLmodel()
        C = haskey(regressors, :C) ? regressors.C : emptyMLmodel()
        n = haskey(regressors, :n) ? regressors.n : emptyMLmodel()
        U = haskey(regressors, :U) ? regressors.U : emptyMLmodel()

        new{typeof(A), typeof(C), typeof(n), typeof(U), typeof(target), typeof(θ)}(A, C, n, U, target, θ)
    end
end
