### Dummy target for testing

function build_target_foo()
    return SIA2D_target(
        :foo,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> 1.0,
        (; H, ∇S, θ, ice_model, ml_model, glacier, params) -> nothing
    )
end