export plot_inverted_fields, inverted_field_panels

# A field on the staggered grid, of size (nx - 1, ny - 1), on the grid of H (NaN in the last row and column)
function _staggered_on_grid(A, shape)
    out = fill(NaN, shape)
    out[1:(end - 1), 1:(end - 1)] .= A
    return out
end

# NaN where the value is not finite, for example log10(0)
finite_or_nan(A) = ifelse.(isfinite.(A), A, NaN)

"""
    inverted_field_panels(inversion::Inversion; glacier_idx = 1) -> Vector{Sleipnir.MapPanel}

The maps of [`plot_inverted_fields`](@ref), as data and without drawing them. Use them to make your
own figure with `Sleipnir.plot_field_maps`.
"""
function inverted_field_panels(inversion::Inversion; glacier_idx::Integer = 1)
    glacier = inversion.glaciers[glacier_idx]
    θ = inversion.model.trainable_components.θ
    physical = inversion.parameters.physical
    shape = size(glacier.H₀)
    H₀ = haskey(θ, :IC) ? initial_thickness(θ, inversion, glacier_idx) : glacier.H₀
    on_cells(A, cells) = ifelse.(cells, A, NaN)
    ice = H₀ .> 0
    ice_staggered = Huginn.avg(H₀) .> 0
    panels = Sleipnir.MapPanel[]
    if haskey(θ, :IC)
        push!(panels, Sleipnir.MapPanel(on_cells(H₀, ice), "Initial ice thickness"; label = "m"))
        push!(panels,
            Sleipnir.MapPanel(
                on_cells(Huginn.bed(inversion, glacier_idx, θ), glacier_cells(glacier)),
                "Bed elevation"; cmap = :terrain, label = "m a.s.l."))
    end
    if haskey(θ, :A)
        θ_A = glacier_params(θ.A, glacier_idx)     # before the broadcast, which would go into θ.A
        A = @. physical.minA + (physical.maxA - physical.minA) * (tanh(θ_A) + 1) / 2
        push!(panels,
            Sleipnir.MapPanel(
                _staggered_on_grid(on_cells(finite_or_nan(log10.(A)), ice_staggered), shape),
                "Rheology A"; label = "log₁₀ A"))
    end
    if haskey(θ, :C)
        θ_C = glacier_params(θ.C, glacier_idx)
        C = @. physical.maxC * (tanh(θ_C) + 1) / 2
        push!(panels,
            Sleipnir.MapPanel(
                _staggered_on_grid(on_cells(finite_or_nan(log10.(C)), ice_staggered), shape),
                "Sliding coefficient C"; label = "log₁₀ C"))
        # The velocity helpers of Huginn read the cache of the glacier, as the training does
        inversion.cache = init_cache(inversion.model, inversion, glacier_idx, θ)
        t₀ = glacier_tspan(inversion, glacier_idx)[1]
        fraction = try
            Huginn.sliding_fraction(H₀, inversion, t₀, θ)
        catch err
            err isa ArgumentError || rethrow()
            nothing                              # a U law gives the total velocity
        end
        isnothing(fraction) || push!(panels,
            Sleipnir.MapPanel(
                _staggered_on_grid(fraction, shape), "Share of sliding in the surface speed";
                label = "sliding / total"))
    end
    isempty(panels) && throw(ArgumentError(
        "The inversion has no gridded parameter (A, C or the initial condition)."))
    return panels
end

"""
    plot_inverted_fields(inversion::Inversion; glacier_idx = 1, title = nothing) -> Figure

Maps of the gridded fields that an inversion found for a glacier, after the training. It shows
what is in `θ`:

  - with the initial condition: the initial ice thickness, and the bed (see [`InitialCondition`](@ref));
  - with a gridded `A`: the rheology `A`;
  - with a gridded `C`: the sliding coefficient `C` and the share of sliding in the surface speed
    (if the laws allow to split the velocity, which a `U` law does not).

Cells without ice are not drawn. The maps use `Sleipnir.plot_field_maps`, with the same style and
rules as the other plots of the inversion (`plot_glacier(results, "fit", ...)`,
`"surface change"` and `"bed"` of `Sleipnir.plot_glacier`): the colour range is
the full range of the data.

`glacier_idx` is the index of the glacier in the inversion.
"""
function plot_inverted_fields(
        inversion::Inversion;
        glacier_idx::Integer = 1,
        title::Union{Nothing, AbstractString} = nothing
)
    isempty(inversion.results.simulation) && throw(ArgumentError(
        "plot_inverted_fields: the inversion has no results. Run the inversion first."))
    panels = inverted_field_panels(inversion; glacier_idx)
    return Sleipnir.plot_field_maps(inversion.results.simulation[glacier_idx], panels;
        ncols = min(length(panels), 2),
        title = something(title, "Inverted fields of $(inversion.glaciers[glacier_idx].rgi_id)"))
end
