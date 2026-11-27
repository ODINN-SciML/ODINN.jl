export Inversion, loss_terms

"""
    mutable struct Inversion{MODEL, CACHE, GLACIER, RES} <: Simulation

An object representing an inversion simulation.
It can involve at the same time a classical inversion and a functional inversion (i.e. the inversion of a function using some data-driven regressor).

# Fields

  - `model::Sleipnir.Model`: The model used for the simulation.
  - `glaciers::Vector{Sleipnir.AbstractGlacier}`: A vector of glaciers involved in the simulation.
  - `parameters::Sleipnir.Parameters`: The parameters used for the simulation.
  - `results::ODINN.Results`: A `ODINN.Results` instance to store the results of the inversion and of the forward simulations.
"""
mutable struct Inversion{
    MODEL <: Sleipnir.Model,
    CACHE,
    GLACIER <: Sleipnir.AbstractGlacier,
    PARAMS <: Sleipnir.Parameters,
    RES <: ODINN.Results
} <: Simulation
    model::MODEL
    cache::Union{CACHE, Nothing}
    glaciers::Vector{GLACIER}
    parameters::PARAMS
    results::RES
end

"""
    function Inversion(
        model::M,
        glaciers::Vector{G},
        parameters::P
    ) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}

Constructor for Inversion struct with glacier model information, glaciers, and parameters.

When the initial condition is trained, a glacier whose first observation (ice thickness survey,
DEM or surface velocity map) is at most one `step_MB` after its start is moved to start on it
(see `align_start!`, which by default starts on the first observation however late it is).
This writes the new window in `parameters.simulation.glacier_tspans`, so it also applies to
other simulations built with the same `parameters`.

# Arguments

  - `model::Sleipnir.Model`: The model used for the simulation.
  - `glaciers::Vector{G}`: A vector of glaciers involved in the simulation.
  - `parameters::Sleipnir.Parameters`: The parameters used for the simulation.

# Returns

  - `Inversion`: A new instance of the Inversion struct.
"""
function Inversion(
        model::M,
        glaciers::Vector{G},
        parameters::P
) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}

    # We perform this check here to avoid having to provide the parameters when creating the model
    @assert targetType(model.trainable_components.target) == parameters.UDE.target "Target does not match the one provided in the parameters."
    Muninn.validate_model_simulation_compatibility(model, parameters)

    # Optionally calibrate the mass balance model per glacier (no-op unless the
    # model type defines a calibration routine, e.g. TImodel1). Calibration returns a new
    # Model since one model per glacier changes the type of the mass_balance field, so the
    # return value has to be used, as Huginn's Prediction does. The calibration uses the
    # geodetic period, not the time window of the glacier, so it can come before `align_start!`.
    if parameters.simulation.use_MB && parameters.simulation.calibrate_MB
        model = calibrate_MB_model(model, glaciers, parameters)
    end

    # Build the results struct based on input values
    emptySimulationResults = Vector{Sleipnir.Results{Sleipnir.Float, Sleipnir.Int}}([])
    emptyResults = Results(emptySimulationResults, TrainingStats())
    inversion = Inversion{
        typeof(model), cache_type(model), G, typeof(parameters), typeof(emptyResults)}(
        model, nothing,
        glaciers,
        parameters,
        emptyResults)

    # When H₀ is trained, a glacier whose first observation is at most one mass balance step
    # from its start starts on it, so the observation constrains H₀ directly.
    # Otherwise, only warn about an observation close to the start.
    check_glacier_tspans(parameters, glaciers)
    align_start!(parameters, glaciers; within = parameters.simulation.step_MB,
        apply = !isnothing(model.trainable_components.θ) &&
                haskey(model.trainable_components.θ, :IC))

    # Fail early with a clear message instead of deep inside the first gradient call
    check_loss(parameters.UDE.empirical_loss_function, inversion)
    θ = model.trainable_components.θ
    # The outline of the glacier constrains H₀: it should be from the date of the start
    !isnothing(θ) && haskey(θ, :IC) && check_outline_date(inversion)
    # A free bed needs the automatic adjoint, which differentiates through the bed, and a DEM
    if any(i -> bed_is_free(θ, i), eachindex(glaciers))
        parameters.UDE.grad isa SciMLSensitivityAdjoint || throw(ArgumentError(
            "The bed is free (`InitialCondition` with `bed = :free`, the default), which \
            needs `SciMLSensitivityAdjoint`. Use `InitialCondition(...; bed = :fixed)` with \
            the manual adjoints."))
        uses_loss(parameters.UDE.empirical_loss_function, LossS) || throw(ArgumentError(
            "The bed is free (`InitialCondition` with `bed = :free`, the default), which \
            needs the surface observations of a `LossS`. Add `LossS()` to the loss, or use \
            `InitialCondition(...; bed = :fixed)`."))
        # With a negative H₀ the bed S₀ - H₀ is above the surface S₀
        parameters.UDE.initial_condition_filter == :identity &&
            @warn "The bed is free with `initial_condition_filter = :identity`, so H₀ can \
                become negative (then the bed S₀ - H₀ is above the surface). Use `:Zang1980` \
                (default) or `:softplus`."
    end

    return inversion
end

# Display setup
Base.show(io::IO, ::MIME"text/plain", inversion::Inversion) = Base.show(io, inversion)
function Base.show(io::IO, inversion::Inversion)
    pad = 14

    println(io, "Inversion")

    # ── glaciers ──────────────────────────────────────────────────────────────
    label(io, "  glaciers", pad)
    n = length(inversion.glaciers)
    val(io, "$n");
    hint(io, " $(n == 1 ? "glacier" : "glaciers")")
    println(io)

    # ── model ─────────────────────────────────────────────────────────────────
    label(io, "  model", pad)
    field(io, "iceflow");
    print(io, " = ")
    val(io, "$(nameof(typeof(inversion.model.iceflow)))")
    sep(io)
    field(io, "mass_balance");
    print(io, " = ")
    val(io, "$(nameof(typeof(inversion.model.mass_balance)))")
    sep(io)
    field(io, "learnable");
    print(io, " =")
    if isnothing(inversion.model.trainable_components)
        hint(io, " (nothing)")
        println(io)
    else
        println(io)
        tc_str = sprint(show, inversion.model.trainable_components)
        for line in split(tc_str, "\n")
            isempty(line) && continue
            printstyled(io, "    "; color = :light_black)
            println(io, line)
        end
    end

    # ── parameters ────────────────────────────────────────────────────────────
    label(io, "  parameters", pad)
    println(io)
    params_str = sprint(show, inversion.parameters)
    for line in split(params_str, "\n")
        isempty(line) && continue
        occursin(r"^Parameters$", line) && continue
        printstyled(io, "    "; color = :light_black)
        println(io, line)
    end

    # ── cache ─────────────────────────────────────────────────────────────────
    label(io, "  cache", pad)
    if isnothing(inversion.cache)
        hint(io, "(nothing)")
    else
        val(io, "$(nameof(typeof(inversion.cache)))")
    end
    println(io)

    # ── results ───────────────────────────────────────────────────────────────
    label(io, "  results", pad)
    stats = inversion.results.stats
    if stats.niter == 0
        print(io, check(false));
        hint(io, " not yet run")
    else
        total_epochs = inversion.parameters.hyper.epochs isa Vector ?
                       sum(inversion.parameters.hyper.epochs) :
                       inversion.parameters.hyper.epochs
        print(io, check(true))
        field(io, " epoch");
        print(io, " = ");
        val(io, "$(stats.niter)")
        hint(io, " / $total_epochs")
        sep(io)
        field(io, "loss");
        print(io, " = ");
        val(io, "$(last(stats.losses))")
        sep(io)
        field(io, "#(simulation)");
        print(io, " = ")
        val(io, "$(length(inversion.results.simulation))")
    end
    println(io)
end

###############################################
################### UTILS #####################
###############################################

include("sciml_utils.jl")
include("inversion_utils.jl")
include("callback_utils.jl")
