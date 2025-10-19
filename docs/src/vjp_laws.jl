# # Law VJP customization

# This tutorial explains how to customize VJP (vector-Jacobian product) computation of the laws in ODINN.jl and clarifies the runtime flow used internally by the library.
# It explains which functions are part of the public, user-facing customization API and which are internal helpers used by ODINN when an automatic-differentiation (AD) backend is required.

# It assumes that you have followed the [Laws](./laws.md) tutorial.

using Revise
using ODINN
rgi_ids = ["RGI60-11.03638"]
rgi_paths = get_rgi_paths()
params = Parameters(
    simulation = SimulationParameters(rgi_paths=rgi_paths),
    UDE = UDEparameters(grad=ContinuousAdjoint()),
)
# params = Parameters()
nn_model = NeuralNetwork(params)

# ## Explanations

# ### High-level summary

# At the user level the customization can be made by implementing hand-written VJPs through the following functions:
# - `f_VJP_input`(...) — VJP w.r.t. inputs
# - `f_VJP_θ`(...) — VJP w.r.t. parameters θ
# - You may also implement your own precompute function to cache expensive computations which is the purpose of `p_VJP`(...). This function is called before solving the adjoint iceflow PDE.

# Internally when the user does NOT provide VJPs, ODINN uses a default AD backend (via [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)).
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
#     - It is invoked when ODINN must fall back to the AD backend (with [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)) because the law did not supply explicit VJP functions (`f_VJP_input`/`f_VJP_θ` or because `p_VJP` is set to `DIVJP()`).
#     - Its job is to precompile and prepare the AD-based VJP code for a given law and to produce *preparation* objects that store [preparation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/explanation/operators/#Preparation) results.
#     - `prepare_vjp_law` is typically called just after the iceflow model / law objects have been instantiated — i.e., early in setup — so that preparations are ready before solving or adjoint runs.

# - `precompute_law_VJP` (used before solving the adjoint PDE)
#   We have overloads in the codebase:
#   ```
#   precompute_law_VJP(
#     law::AbstractLaw,
#     cache,
#     vjpsPrepLaw::AbstractPrepVJP,
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
# How the pieces compose in practice?
# - If you, as a user, provide custom VJP functions (through `f_VJP_input!`/`f_VJP_θ!`, or through `p_VJP!`), ODINN will use them directly at adjoint time and will skip the AD fallback path. You can also provide your own precompute wrapper and cache to optimize expensive computations.
# - If you do NOT provide VJP functions, ODINN runs the AD fallback:
#   1. `prepare_vjp_law` runs early (post-instantiation) to compile/prepare AD-based helpers and returns some `AbstractPrepVJP` object.
#   2. `precompute_law_VJP` is skipped.
#   3. During the adjoint solve, `law_VJP_input` and `law_VJP_θ` use the preparation objects precompiled in `prepare_vjp_law` to automatically differentiate `f!` with [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) and obtain the VJPs of the law with respect to the inputs and to the parameters `θ`.

# ### User level customization
# What is user-visible and can be customized?
#   - `f_VJP_input!(cache, inputs, θ)` — compute the VJP with respect to the inputs and store the result in `cache.vjp_inp`
#   - `f_VJP_θ!(cache, inputs, θ)` — compute the VJP with respect to `θ` and store the result in `cache.vjp_θ`
#   - `p_VJP!(cache, vjpsPrepLaw, inputs, θ)` — if you want to precompute some components (or even the whole VJPs when possible) before solving the adjoint iceflow PDE
#   - custom cache implementations (described below)

# ### Notes on cache definition
# The `cache` parameter that flows through `p_VJP!`/`f_VJP_*` calls is the place to store artifacts useful for efficient computation as well as the results of the VJPs computation.
# It following fields are mandatory:
# - `value`: a placeholder to store the result of the forward evaluation, can be of any type
# - `vjp_θ::Vector{Float64}`: a placeholder to store the result of the VJP with respect to `θ`, must be a vector
# - `vjp_inp`: a placeholder to store the result of the VJP with respect to the inputs, must be of a type that matches the one the inputs are defined in
# In order to know the type of the inputs, simply run `generate_inputs(law.f.inputs, simulation, glacier_idx, t)`.

# ### Use the preparation object

# ### Best practices and debugging tips
# - If you supply custom VJPs, test them with finite-difference checks for both inputs and parameters. ODINN does not check that the correctness of your implementation!
# - If you rely on ODINN's AD fallback, be aware that `prepare_vjp_law` will precompile and prepare AD helpers at model instantiation time — expect longer setup time but faster adjoint runs thereafter.
# - Inspect/validate cache content if you get inconsistent adjoints — a stale or incorrect cache entry is a common cause.
# - Although the API is designed to provide every thing you need as arguments, if your VJP needs anything from the forward pass, ensure it is stored in the cache.


# ## Simple VJP customization

# We will explore how we can customize the VJP computation of the law that is used in the [Laws](./laws.md) tutorial.
# The cache used for this law is a `ScalarCache` since the output of this law is a scalar value `A`, the creep coefficient.
# We can confirm that this type defines the fields needed for the VJP computation:

fieldnames(ScalarCache)

# We retrieve the model architecture, the physical parameters to be used inside the `f!` of the law and we define the inputs:

archi = nn_model.architecture
st = nn_model.st
smodel = ODINN.StatefulLuxLayer{true}(archi, nothing, st)
min_NN = params.physical.minA
max_NN = params.physical.maxA
inputs = (; T=InpTemp())

# We will also need `θ` in order to call the VJPs of the law manually although in practice we do not have to worry about retrieving this:

θ = nn_model.θ

# And then the `f!` and `init_cache` functions:
f! = let smodel = smodel, min_NN = min_NN, max_NN = max_NN
    function (cache, inp, θ)
        inp = collect(values(inp))
        A = only(scale(smodel(inp, θ.A), (min_NN, max_NN)))
        ODINN.Zygote.@ignore cache.value .= A # We ignore this in-place affectation in order to be able to differentiate it with Zygote hereafter
        return A
    end
end
function init_cache(simulation, glacier_idx, θ; scalar=false)
    return ScalarCache(zeros(), zeros(), zero(θ))
end


# The declaration of the law without VJP customization would be:

law = Law{ScalarCache}(;
    inputs = inputs,
    f! = f!,
    init_cache = init_cache,
)

# We see from the output that the VJPs are inferred using [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) and that ODINN does not use precomputation.
# Now let's try to customize the VJPs by manually implementing the AD step:

law = Law{ScalarCache}(;
    inputs = inputs,
    f! = f!,
    f_VJP_input! = function (cache, inputs, θ)
        nothing # The input does not depend on the glacier state
    end,
    f_VJP_θ! = function (cache, inputs, θ)
        @show inputs
        # ODINN.Zygote.gradient(_θ -> f!(cache, inputs))
    end,
    init_cache = init_cache,
)
# law_cache = 

# In order to instantiate the cache, we need to define the model:

rgi_ids = ["RGI60-11.03638"]
glaciers = initialize_glaciers(rgi_ids, params)
model = Model(
    iceflow = SIA2Dmodel(params; A=law),
    mass_balance = nothing,
    regressors = (; A=nn_model)
)
simulation = FunctionalInversion(model, glaciers, params)
θ = simulation.model.machine_learning.θ
glacier_idx = 1
simulation.cache = ODINN.init_cache(model, simulation, glacier_idx, θ)

vjpsPrepLaw = nothing
backend = nothing
# backend=params.UDE.grad.VJP_method.regressorADBackend
# ∂law∂θ!(backend, law, simulation.cache.iceflow.A, vjpsPrepLaw, (; T=1.0), θ)
ODINN.∂law∂θ!(backend, simulation.model.iceflow.A, simulation.cache.iceflow.A, simulation.cache.iceflow.A_prep_vjps, (; T=1.0), θ)

# ## Simple cache customization





#   2. `precompute_law_VJP` is called before solving the adjoint PDE for a specific time `t` (and θ). It ensures the `cache` contains whatever precomputed objects are needed for efficient adjoint solves.

# explain how we can use the prep object in the precompute function

# Add FAQ