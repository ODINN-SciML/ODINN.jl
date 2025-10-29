"""
The goal of this inversion is to recover the power law dependency of the diffusivity in
the SIA equation as a function of H.

We are going go generate a glacier flow equation following the next SIA diffusivity:

D(H, ∇S) = 2A / (n + 2) * (ρg)^n H^{n+2} |∇S|^{n-1}

and we are going to target this diffusivity with the function

D(H, ∇S, θ) = H * NN(Temp, H, ∇S)

The neural network should learn the function NN(Temp, H, ∇S) ≈  2A / (n + 2) * (ρg)^n H^{n+1} |∇S|^{n-1}
which corresponds to the time integrated ice surface velocity.

with n the Glen exponent.

In order to evaluate the quality of our inversions, we are going to use analytical solutions
given by Halfar solutions to evaluate the performance of the invesion.
"""

using Pkg
# Activate the "scripts" environment, this works both if the user is in "ODINN/", in "ODINN/scripts/" or in any subfolder
odinn_folder = split(Base.source_dir(), "scripts")[1]
Pkg.activate(odinn_folder*"/scripts/")
Pkg.develop(Pkg.PackageSpec(path = odinn_folder)) # Set ODINN in dev mode to use local version, you might do as well for Huginn, Muninn and Sleipnir

using Revise
using ODINN
using Sleipnir: DummyClimate2D
# using SciMLSensitivity
# using Lux, ComponentArrays
# using Statistics
# using Plots
# using JLD2
# using Random, Distributions
using ODINN: MersenneTwister, ZygoteAdjoint, ComponentVector
using ODINN: Lux, sigmoid, gelu, softplus

rng = MersenneTwister(616)

rgi_paths = get_rgi_paths()

# The value of this does not really matter, it is hardcoded in Sleipnir right now.
working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")

use_MB = false
λ = use_MB ? 5.0 : 0.0
nx, ny = 100, 100
R₀ = 2000.0
H₀ = 200.0
H_max = 1.2 * H₀
A₀ = 2e-18
n₀ = 3.0

halfar_params = HalfarParameters(λ = λ, R₀ = R₀, H₀ = H₀, A = A₀, n = n₀)
halfar, t₀ = Halfar(halfar_params)

Δt = 30.0
t₁ = t₀ + Δt
δt = Δt / 200
tstops = Huginn.define_callback_steps((t₀, t₁), δt) |> collect

B = zeros((nx,ny))
# Construct a grid that includes the initial Dome
η = 0.66
Δx = R₀ / nx / (η / 2)
Δy = R₀ / ny / (η / 2)
xs = [(i - nx / 2) * Δx for i in 1:nx]
ys = [(j - ny / 2) * Δy for j in 1:ny]

# Construct analytical time series
Hs = [[halfar(x, y, t) for x in xs, y in ys] for t in tstops]

params = Parameters(
    simulation = SimulationParameters(
        working_dir = working_dir,
        use_MB = use_MB,
        use_velocities = true,
        tspan = (t₀, t₁),
        step = δt,
        multiprocessing = false,
        workers = 1,
        test_mode = false,
        rgi_paths = rgi_paths,
        gridScalingFactor = 4,
        ),
    hyper = Hyperparameters(
        batch_size = 1,
        epochs = [100, 30],
        optimizer = [
            ODINN.ADAM(0.001),
            ODINN.LBFGS(
                linesearch = ODINN.LineSearches.BackTracking(iterations = 10)
                )
                # ODINN.LineSearches.HagerZhang( # See https://github.com/JuliaNLSolvers/LineSearches.jl/blob/3259cd240144b96a5a3a309ea96dfb19181058b2/src/hagerzhang.jl#L37
                #     linesearchmax = 10,
                #     display = true,
                #     delta = 0.01,
                #     sigma = 0.1)
                #     )
                ]
        ),
    UDE = UDEparameters(
        sensealg = ZygoteAdjoint(),
        optim_autoAD = ODINN.NoAD(),
        grad = ContinuousAdjoint(),
        optimization_method = "AD+AD",
        target = :D
        ),
    solver = Huginn.SolverParameters(
        step = δt,
        save_everystep = true,
        progress = true
        )
    )

# We are going to create a glacier using the Halfar solution
glacier = Glacier2D(
    rgi_id = "Halfar",
    climate = DummyClimate2D(),
    H₀ = Hs[begin],
    S = B + Hs[begin],
    B = B,
    A = halfar_params.A,
    n = halfar_params.n,
    Δx = Δx,
    Δy = Δy,
    nx = nx,
    ny = ny,
    C = 0.0
    )
glaciers = Vector{Sleipnir.AbstractGlacier}([glacier])

# We add thickness data to Glacier object
glaciers[1] = Glacier2D(glaciers[1], thicknessData = Sleipnir.ThicknessData(tstops, Hs))


"""
We can define the architecture of the model directly, passing the prescale and postcale
directly to Lux using a WrappedFunction layer.
"""
# n_fourier_feautures = 10

function inv_normalize(v::Union{Vector,SubArray})
    @assert length(v) == 2
    return [ODINN.normalize(v[1]; lims = (0.0, H_max)), ODINN.normalize(v[2]; lims = (0.0, 0.6))]
end

# function inv_normalize(V::Matrix)
#     @assert size(V)[1] == 2
#     M = reduce(hcat, map(v -> inv_normalize(v), eachcol(V)))
#     return M
# end

function inv_fourier_feature(v::Union{Vector,SubArray})
    return [fourier_feature(v[1], n_fourier_feautures); fourier_feature(v[2], n_fourier_feautures)]
end

# function inv_fourier_feature(V::Matrix)
#     M = reduce(hcat, map(v -> inv_fourier_feature(v), eachcol(V)))
#     return M
# end

architecture = Lux.Chain(
    Lux.WrappedFunction(x -> LuxFunction(inv_normalize, x)),
    # WrappedFunction(x -> inv_fourier_feature(x)),
    # WrappedFunction(x -> [fourier_feature(x[1], n_fourier_feautures); fourier_feature(x[2], n_fourier_feautures)]),
    Lux.Dense(2, 5, x -> gelu.(x)),
    Lux.Dense(5, 8, x -> gelu.(x)),
    Lux.Dense(8, 20, x -> gelu.(x)),
    # Dense(4 * n_fourier_feautures, 20, x -> softplus.(x)),
    Lux.Dense(20, 30, x -> softplus.(x)),
    Lux.Dense(30, 10, x -> softplus.(x)),
    Lux.Dense(10, 1, sigmoid),
    Lux.WrappedFunction(y -> 1e5 .* exp.((y .- 1.0) ./ y))
    # WrappedFunction(y -> 10.0.^( 3.0 .* (y .- 1.0) .+ 1.0 .* (y .+ 1.0) ))
)

### Pretraining

H_samples = 0.0:2.0:H_max
∇S_samples = 0.0:0.02:0.4
all_combinations = vec(collect(Iterators.product(H_samples, ∇S_samples)))
X_samples = hcat(first.(all_combinations), last.(all_combinations))
function template_diffusivity(h, ∇s)
    (; ρ, g) = params.physical
    n = 1.01 * n₀
    A = 0.80 * A₀
    # This one has one less H than the actual diffusivity
    return 2 * A * (ρ * g)^n * h^(n+1) * ∇s^(n-1) / (n + 2)
end
Y_samples = map(x -> template_diffusivity(x[1], x[2]), eachrow(X_samples))
# Set matrices in right format
X_samples = transpose(X_samples)
X_samples = Matrix(X_samples)
Y_samples = reshape(Y_samples, (1, :))

architecture, θ_pretrain, st_pretrain, losses = pretraining(
    architecture;
    X = X_samples, Y = Y_samples,
    nepochs = 5000, rng = rng
)

# We define the prescale and postscale of quantities.
nn_model = NeuralNetwork(
    params;
    architecture = architecture,
    θ = ComponentVector(θ = θ_pretrain), # We should give the actual solution!!!
    seed = rng
)
model = Model(
    iceflow = SIA2Dmodel(params; U=LawU(nn_model, params; prescale_bounds=nothing, max_NN=nothing)),
    mass_balance = TImodel1(
        params; DDF = 6.0/1000.0,
        acc_factor = 1.2/1000.0
        ),
    regressors = (; U=nn_model),
    target = SIA2D_D_target(
        interpolation = :Linear,
        n_interp_half = 5, # Notice we use a very low value for this!
    ),
)

# We create an ODINN prediction
functional_inversion = Inversion(model, glaciers, params)
