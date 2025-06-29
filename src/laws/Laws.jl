import Sleipnir: get_input, default_name

export LawA, LawDhybrid, LawU

function _pred_NN(inp::Vector{F}, smodel, θ, prescale, postscale) where {F <: AbstractFloat}
    return only(postscale(smodel(prescale(inp), θ)))
end

function LawU(
    nn_model,
    params;
    max_NN = 50.0,
    prescale_bounds = [(0.0, 300.0), (0.0, 0.5)],
    use_postscale = true,
)
    use_prescale = !isnothing(prescale_bounds)
    # This shouuld correspond to maximum of Umax * dSdx
    max_NN = isnothing(max_NN) ? params.physical.maxA : max_NN
    prescale = use_prescale ? X -> _ml_model_prescale(X, prescale_bounds) : identity
    postscale = use_postscale ? Y -> _ml_model_postscale(Y, max_NN) : identity

    archi = nn_model.architecture
    st = nn_model.st
    smodel = StatefulLuxLayer{true}(archi, nothing, st)

    A_law = let smodel = smodel, prescale = prescale, postscale = postscale
    Law{Array{Float64, 2}}(;
        inputs = (; H̄=InpH̄(), ∇S=Inp∇S()), # TODO: define scaling in inputs
        f! = function (cache, inp, θ)
            D = ((h, ∇s) -> _pred_NN([h, ∇s], smodel, θ.U, prescale, postscale)).(inp.H̄, inp.∇S)

            # Flag the in-place assignment as non differented and return D instead in
            # order to be able to compute ∂D∂θ with Zygote
            Zygote.@ignore cache .= D
            return D
        end,
        init_cache = function (simulation, glacier_idx, θ)
            (; nx, ny) = simulation.glaciers[glacier_idx]
            return zeros(nx-1, ny-1)
        end,
    )
    end

    return A_law
end


function LawDhybrid(
    nn_model,
    params;
    max_NN = nothing,
    prescale_bounds = [(-25.0, 0.0), (0.0, 500.0)],
    use_postscale = true,
)
    use_prescale = !isnothing(prescale_bounds)
    max_NN = isnothing(max_NN) ? params.physical.maxA : max_NN
    prescale = use_prescale ? X -> _ml_model_prescale(X, prescale_bounds) : identity
    postscale = use_postscale ? Y -> _ml_model_postscale(Y, max_NN) : identity

    archi = nn_model.architecture
    st = nn_model.st
    smodel = StatefulLuxLayer{true}(archi, nothing, st)

    A_law = let smodel = smodel, prescale = prescale, postscale = postscale
    Law{Array{Float64, 2}}(;
        inputs = (; T=InpTemp(), H̄=InpH̄()),
        f! = function (cache, inp, θ)
            A = map(h -> _pred_NN([inp.T, h], smodel, θ.A, prescale, postscale), inp.H̄)

            # Flag the in-place assignment as non differented and return A instead in
            # order to be able to compute ∂A∂θ with Zygote
            Zygote.@ignore cache .= A
            return A
        end,
        init_cache = function (simulation, glacier_idx, θ)
            (; nx, ny) = simulation.glaciers[glacier_idx]
            return zeros(nx-1, ny-1)
        end,
    )
    end

    return A_law
end


function LawA(
    nn_model,
    params,
)
    archi = nn_model.architecture
    st = nn_model.st
    smodel = StatefulLuxLayer{true}(archi, nothing, st)

    A_law = let smodel = smodel, params = params
        Law{Array{Float64, 0}}(;
            inputs = (; T=InpTemp()),
            f! = function (cache, inp, θ)
                min_NN = params.physical.minA
                max_NN = params.physical.maxA
                inp = collect(values(inp))
                A = only(scale(smodel(inp, θ.A), (min_NN, max_NN)))

                # Flag the in-place assignment as non differented and return A instead in
                # order to be able to compute ∂A∂θ with Zygote
                Zygote.@ignore cache .= A
                return A
            end,
            init_cache = function (simulation, glacier_idx, θ)
                return zeros()
            end,
        )
    end

    return A_law
end

include("laws_utils.jl")
