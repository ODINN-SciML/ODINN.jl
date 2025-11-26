# # Law VJP customization

# This tutorial explains how to customize VJP (vector-Jacobian product) computation of the laws in ODINN.jl and clarifies the runtime flow used internally by the library.
# It explains which functions are part of the public, user-facing customization API and which are internal helpers used by ODINN when an automatic-differentiation (AD) backend is required.

# It assumes that you have followed the [Laws](./laws.md) tutorial.

using ODINN
rgi_ids = ["RGI60-11.03638"]
rgi_paths = get_rgi_paths()
params = Parameters(
    simulation = SimulationParameters(rgi_paths=rgi_paths),
    UDE = UDEparameters(grad=ContinuousAdjoint()),
)
nn_model = NeuralNetwork(params)

# ## Explanations

# ### High-level summary

# At the user level the customization can be made by implementing hand-written VJPs through the following functions:
# - `f_VJP_input!(...)` — VJP w.r.t. inputs
# - `f_VJP_θ!(...)` — VJP w.r.t. parameters θ
# - You may also implement your own precompute function to cache expensive computations which is the purpose of `p_VJP!(...)`. This function is called before solving the adjoint iceflow PDE.

# Internally when the user does NOT provide VJPs, ODINN uses a default AD backend (via [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)) to compute the VJPs of the laws.
# To support efficient reverse-mode execution, ODINN will:
#   - compile and precompute adjoint-related helper functions and
#   - store preparation objects that are used later during the in adjoint PDE.
# This mechanism is triggered by `prepare_vjp_law`.

# ### Internal function roles
# - `prepare_vjp_law` (internal)

#   Signature used in the codebase:
#   ```
#   prepare_vjp_law(
#       simulation,
#       law::AbstractLaw,
#       law_cache,
#       θ,
#       glacier_idx,
#   )
#   ```
#   - Intent and behavior:
#     - This is an internal routine. It is NOT intended to be called by users directly.
#     - It is invoked when ODINN must fall back to the AD backend (with [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)) because the law did not supply explicit VJP functions (`f_VJP_input!`/`f_VJP_θ!` or because `p_VJP!` is set to `DIVJP()`).
#     - Its job is to precompile and prepare the AD-based VJP code for a given law and to produce *preparation* objects that store [preparation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/explanation/operators/#Preparation) results.
#     - `prepare_vjp_law` is typically called just after the iceflow model / law objects have been instantiated — i.e., early in the setup — so that preparations are ready before solving or adjoint runs.

# - `precompute_law_VJP` (used before solving the adjoint PDE)

#   The typical signature in the codebase is:
#   ```
#   precompute_law_VJP(
#     law::AbstractLaw,
#     cache,
#     vjpsPrepLaw,
#     simulation,
#     glacier_idx,
#     t,
#     θ
#   )
#   ```
#   - Intent and behavior:
#     - This function precomputes VJP-related artifacts *before* the adjoint iceflow PDE is solved for given time `t` and parameters `θ`.
#     - It typically uses the `vjpsPrepLaw` (an `AbstractPrepVJP` instance produced earlier by `prepare_vjp_law`) together with the `cache` and `simulation` object. The produced results are cached in `cache` and are optionally consumed later by `law_VJP_input` / `law_VJP_θ` during the adjoint solve.

# - Entry points used in the adjoint PDE

#   These functions are the actual runtime entry points used when computing contributions of the laws to the gradient in the adjoint PDE:
#   ```
#   law_VJP_θ(law::AbstractLaw, cache, simulation, glacier_idx, t, θ)
#   ```
#   and
#   ```
#   law_VJP_input(law::AbstractLaw, cache, simulation, glacier_idx, t, θ)
#   ```
#   - Intent and behavior:
#     - These are called during the adjoint solve to compute parameter and input VJPs for the law at time `t` and for parameters `θ`.
#     - They can either compute the VJPs directly or use cached VJP information that has been already computed in the user-supplied `p_VJP!` VJP function. The `cache` allows storing useful information from the forward or from the precomputation step.
#     - They therefore carry the runtime context (simulation, glacier index, time, θ) which is necessary for adjoint calculations.

# ### Workflow
# For the wide audience we do not recommend to play with the VJPs.
# ODINN comes with default parameters and the average user does not need to customize the VJPs.
# Keeping the default values will work fine.

# Advanced users seeking maximum performance can customize the VJPs which can significantly speed-up the code.

# How do the pieces compose in practice?
# - If you, as a user, provide custom VJP functions (through `f_VJP_input!`/`f_VJP_θ!`, or through `p_VJP!`), ODINN will use them directly at adjoint time and will skip the AD fallback path. You can also provide your own precompute wrapper and cache to optimize expensive computations.
# - If you do NOT provide VJP functions, ODINN runs the AD fallback:
#   1. `prepare_vjp_law` runs early (post-instantiation) to compile/prepare AD-based helpers and returns some `AbstractPrepVJP` object.
#   2. `precompute_law_VJP` is skipped.
#   3. During the adjoint solve, `law_VJP_input` and `law_VJP_θ` use the preparation objects precompiled in `prepare_vjp_law` to automatically differentiate `f!` with [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) and obtain the VJPs of the law with respect to the inputs and to the parameters `θ`.

# !!! info
#     You can change the default AD backend for laws that do not have custom VJPs in the VJP type, for example by setting `VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote())` when you define the adjoint method.

# ### User level customization
# What is user-visible and can be customized?
#   - `f_VJP_input!(cache, inputs, θ)` — compute the VJP with respect to the inputs and store the result in `cache.vjp_inp`
#   - `f_VJP_θ!(cache, inputs, θ)` — compute the VJP with respect to `θ` and store the result in `cache.vjp_θ`
#   - `p_VJP!(cache, vjpsPrepLaw, inputs, θ)` — if you want to precompute some components (or even the whole VJPs when possible) before solving the adjoint iceflow PDE
#   - custom cache implementations (described below)

# ### Notes on cache definition
# The `cache` parameter that is threaded through `p_VJP!`/`f_VJP_*` calls is the place to store artifacts useful for efficient computation as well as the results of the VJPs computation.
# The following fields are mandatory:
# - `value`: a placeholder to store the result of the forward evaluation, can be of any type
# - `vjp_θ`: a placeholder to store the result of the VJP with respect to `θ`, depending on the type of law that is defined, it can be a vector or a 3 dimensional array
# - `vjp_inp`: a placeholder to store the result of the VJP with respect to the inputs, must be of a type that matches the one of the inputs
# In order to know the type of the inputs, simply run `generate_inputs(law.f.inputs, simulation, glacier_idx, t)`.

# ### Using the preparation object

# !!! error
#     For the moment, using the preparation object at the user level is not supported yet.

# ### Best practices and debugging tips
# - If you supply custom VJPs, test them with finite-difference checks for both inputs and parameters. ODINN does not check that the correctness of your implementation!
# - If you rely on ODINN's AD fallback, be aware that `prepare_vjp_law` will precompile and prepare AD helpers at model instantiation time — expect longer setup time but faster adjoint runs thereafter.
# - Inspect/validate cache content if you get inconsistent adjoints — a stale or incorrect cache entry is a common cause.
# - Although the API is designed to provide everything you need as arguments, if your VJP needs anything from the forward pass, ensure it is stored in the cache.


# ## Simple VJP customization

# We will explore how we can customize the VJP computation of the law that is used in the [Laws](./laws.md) tutorial.
# The cache used for this law is a `ScalarCache` since the output of this law is a scalar value `A`, the creep coefficient.
# We can confirm that this type defines the fields needed for the VJP computation:

fieldnames(ScalarCache)

# Before defining the law, we retrieve the model architecture, the physical parameters to be used inside the `f!` function of the law and we define the inputs:

archi = nn_model.architecture
st = nn_model.st
smodel = ODINN.StatefulLuxLayer{true}(archi, nothing, st)
min_NN = params.physical.minA
max_NN = params.physical.maxA
inputs = (; T=iAvgScalarTemp())

# And then the `f!` and `init_cache` functions:
f! = let smodel = smodel, min_NN = min_NN, max_NN = max_NN
    function (cache, inp, θ)
        inp = collect(values(inp))
        A = only(ODINN.scale(smodel(inp, θ.A), (min_NN, max_NN)))
        ODINN.Zygote.@ignore_derivatives cache.value .= A # We ignore this in-place affectation in order to be able to differentiate it with Zygote hereafter
        return A
    end
end
function init_cache(simulation, glacier_idx, θ)
    return ScalarCache(zeros(), zeros(), zero(θ))
end


# The declaration of the law without VJP customization would be:

law = Law{ScalarCache}(;
    inputs = inputs,
    f! = f!,
    init_cache = init_cache,
)

# !!! success
#     We see from the output that the VJPs are inferred using [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) and that ODINN does not use precomputation.
#     Now let's try to customize the VJPs by manually implementing the AD step:

law = Law{ScalarCache}(;
    inputs = inputs,
    f! = f!,
    f_VJP_input! = function (cache, inputs, θ)
        nothing # The input does not depend on the glacier state
    end,
    f_VJP_θ! = function (cache, inputs, θ)
        cache.vjp_θ .= ones(length(θ)) # The VJP is wrong on purpose to check that this function is properly called hereafter
    end,
    init_cache = init_cache,
)

# In order to instantiate the cache, we need to define the model:

rgi_ids = ["RGI60-11.03638"]
glaciers = initialize_glaciers(rgi_ids, params)
model = Model(
    iceflow = SIA2Dmodel(params; A=law),
    mass_balance = nothing,
    regressors = (; A=nn_model)
)
simulation = Inversion(model, glaciers, params)

# We will also need `θ` in order to call the VJPs of the law manually although in practice we do not have to worry about retrieving this:

θ = simulation.model.machine_learning.θ

# We then create the cache, and again all of this is handled internally in ODINN. We need to instantiate manually here to demonstrate how the VJPs can be customized.

glacier_idx = 1
simulation.cache = ODINN.init_cache(model, simulation, glacier_idx, θ)

# Finally we demonstrate that this is our custom implementation that is being called:

ODINN.∂law∂θ!(
    params.UDE.grad.VJP_method.regressorADBackend,
    simulation.model.iceflow.A,
    simulation.cache.iceflow.A,
    simulation.cache.iceflow.A_prep_vjps,
    (; T=1.0), θ)

# ## VJP precomputation

# Since the law that we have been using so far does not depend on the glacier state, it could be computed once for all at the beginning of the simulation and the VJPs could be precomputed before solving the adjoint iceflow PDE.
# The definition of the law below illustrates how we can do this in two ways:
# - by using [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) to automatically compute the VJPs;
# - by manually precomputing the VJPs in the `p_VJP` function.

# ### Automatic precomputation with DI

law = Law{ScalarCache}(;
    inputs = inputs,
    f! = f!,
    init_cache = init_cache,
    p_VJP! = DIVJP(),
    callback_freq = 0,
)

# !!! success
#     This law is applied only once before the beginning of the simulation, and the VJP are precomputed automatically.

# ### Manual precomputation

law = Law{ScalarCache}(;
    inputs = inputs,
    f! = f!,
    init_cache = init_cache,
    p_VJP! = function (cache, vjpsPrepLaw, inputs, θ)
        cache.vjp_θ .= ones(length(θ))
    end,
    callback_freq = 0,
)

# !!! success
#     This law is applied only once before the beginning of the simulation, and the VJP are precomputed using our own implementation.


# ## Simple cache customization

# In this last section we illustrate how we can define our own cache to store additional information.
# Our use case is the interpolation of the VJP on a coarse grid.
# By coarse grid we mean that in order to evaluate the VJP we do not need the differentiate the law for every value of ice thickness we have on the 2D grid at each time step.
# We only need to pre-evaluate the VJP for a few values of H (this set of values corresponds to the coarse grid), and then we can interpolate the precomputed VJP at the required values of H.
# The VJPs on the coarse grid are precomputed before solving the adjoint PDE and the evaluation at the exact points in the adjoint PDE are made using an interpolator that is stored inside the cache object.

params = Parameters(
    simulation = SimulationParameters(rgi_paths=rgi_paths),
    UDE = UDEparameters(grad=ContinuousAdjoint(),
    target = :D_hybrid),
)
nn_model = NeuralNetwork(params)

prescale_bounds = [(-25.0, 0.0), (0.0, 500.0)]
prescale = X -> ODINN._ml_model_prescale(X, prescale_bounds)
postscale = Y -> ODINN._ml_model_postscale(Y, params.physical.maxA)

archi = nn_model.architecture
st = nn_model.st
smodel = ODINN.StatefulLuxLayer{true}(archi, nothing, st)

inputs = (; T=iAvgScalarTemp(), H̄=iH̄())

f! = let smodel = smodel, prescale = prescale, postscale = postscale
    function (cache, inp, θ)
        Y = map(h -> ODINN._pred_NN([inp.T, h], smodel, θ.Y, prescale, postscale), inp.H̄)
        ODINN.Zygote.@ignore_derivatives cache.value .= Y # # We ignore this in-place affectation in order to be able to differentiate it with Zygote hereafter
        return Y
    end
end

# We define a new cache struct to store the interpolator:

using Interpolations
mutable struct MatrixCacheInterp <: Cache
    value::Array{Float64, 2}
    vjp_inp::Array{Float64, 2}
    vjp_θ::Array{Float64, 3}
    interp_θ::Interpolations.GriddedInterpolation{Vector{Float64}, 1, Vector{Vector{Float64}}, Interpolations.Gridded{Interpolations.Linear{Interpolations.Throw{OnGrid}}}, Tuple{Vector{Float64}}}
end

# !!! warning
#     The `cache` must of concrete type.

function init_cache_interp(simulation, glacier_idx, θ)
    glacier = simulation.glaciers[glacier_idx]
    (; nx, ny) = glacier
    H_interp = ODINN.create_interpolation(glacier.H₀; n_interp_half = simulation.model.machine_learning.target.n_interp_half)
    θvec = ODINN.ComponentVector2Vector(θ)
    grads = [zero(θvec) for i in 1:length(H_interp)]
    grad_itp = interpolate((H_interp,), grads, Gridded(Linear()))
    return MatrixCacheInterp(zeros(nx-1, ny-1), zeros(nx-1, ny-1), zeros(nx-1, ny-1, length(θ)), grad_itp)
end

# In order to initialize the cache, we created a fake interpolation grid above.
# However, this interpolation grid will be computed during the precomputation step based on the provided inputs at the beginning of the adjoint PDE.

# Below we define the precomputation function which defines a coarse grid and differentiates the neural network at each of these points.

function p_VJP!(cache, vjpsPrepLaw, inputs, θ)
    H_interp = ODINN.create_interpolation(inputs.H̄; n_interp_half = simulation.model.machine_learning.target.n_interp_half)
    grads = Vector{Float64}[]
    for h in H_interp
        ret, = ODINN.Zygote.gradient(_θ -> f!(cache, (; T=inputs.T, H̄=h), _θ), θ)
        push!(grads, ODINN.ComponentVector2Vector(ret))
    end
    cache.interp_θ = interpolate((H_interp,), grads, Gridded(Linear()))
end

# Then at each iteration of the adjoint PDE, we use the interpolator that we evaluate with the values in `inputs.H̄`.
# Since many of the points are zeros, we evaluate the interpolator for `H̄=0` only once.

function f_VJP_θ!(cache, inputs, θ)
    H̄ = inputs.H̄
    zero_interp = cache.interp_θ(0.0)
    for i in axes(H̄, 1), j in axes(H̄, 2)
        cache.vjp_θ[i, j, :] = map(h -> ifelse(h == 0.0, zero_interp, cache.interp_θ(h)), H̄[i, j])
    end
end

# Finally we can define the law:

law = Law{MatrixCacheInterp}(;
    inputs = inputs,
    f! = f!,
    init_cache = init_cache_interp,
    p_VJP! = p_VJP!,
    f_VJP_θ! = f_VJP_θ!,
    f_VJP_input! = function (cache, inputs, θ) # Not implemented in this example
    end,
)

# As in the previous example, we need to define some objects and make the initialization manually to be able to call the internals of ODINN `ODINN.precompute_law_VJP` and `ODINN.∂law∂θ!`.

rgi_ids = ["RGI60-11.03638"]
glaciers = initialize_glaciers(rgi_ids, params)
model = Model(
    iceflow = SIA2Dmodel(params; Y=law),
    mass_balance = nothing,
    regressors = (; Y=nn_model)
)
simulation = Inversion(model, glaciers, params)
θ = simulation.model.machine_learning.θ
glacier_idx = 1
t = simulation.parameters.simulation.tspan[1]
simulation.cache = ODINN.init_cache(model, simulation, glacier_idx, θ)

# Apply once to be able to retrieve the inputs
dH = zero(simulation.cache.iceflow.H)
ODINN.Huginn.SIA2D!(dH, simulation.cache.iceflow.H, simulation, t, θ);

# Finally we call the precompute function and the VJP function called at each iteration of the adjoint PDE.

ODINN.precompute_law_VJP(
    simulation.model.iceflow.Y,
    simulation.cache.iceflow.Y,
    simulation.cache.iceflow.Y_prep_vjps,
    simulation,
    glacier_idx, t, θ)

ODINN.∂law∂θ!(
    params.UDE.grad.VJP_method.regressorADBackend,
    simulation.model.iceflow.Y,
    simulation.cache.iceflow.Y,
    simulation.cache.iceflow.Y_prep_vjps,
    (; T=1.0, H̄=simulation.cache.iceflow.H̄), θ)

# Now let us check that the `vjp_θ` field of the cache, which is spatially varying, has been populated:

simulation.cache.iceflow.Y.vjp_θ


# ## Frequently Asked Questions
# - Can I use the preparation object in the `p_VJP!`/`f_VJP_*` functions?
# No it is not possible for the moment to use the preparation object inside these functions.
# The preparation object is used to store things precompiled by [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) when `p_VJP!=DIVJP()` and hence it excludes from using it in `p_VJP!`.
# As for `f_VJP_*`, the preparation object cannot be accessed for the moment.
# If there is a need, we might add it as an argument in the future.
