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
using ODINN: MersenneTwister, ZygoteAdjoint, ComponentVector
using ODINN: Lux, sigmoid, gelu, softplus
using JLD2

rng = MersenneTwister(616)

rgi_paths = get_rgi_paths()

# The value of this does not really matter, it is hardcoded in Sleipnir right now.
working_dir = joinpath(homedir(), ".OGGM/ODINN_tests")

use_MB = false
λ = use_MB ? 5.0 : 0.0
nx, ny = 60, 60
# nx, ny = 20, 20
R₀ = 2000.0
H₀ = 400.0
H_max = 1.2 * H₀
A₀ = 1.1e-17 # associated to ice at T ≈ -10C
n₀ = 3.0

halfar_params = HalfarParameters(λ = λ, R₀ = R₀, H₀ = H₀, A = A₀, n = n₀)
halfar, t₀ = Halfar(halfar_params)
halfar_velocity = Huginn.Halfar_velocity(halfar_params)

Δt = 10.0
t₁ = t₀ + Δt
δt = Δt / 200
tstops = Huginn.define_callback_steps((t₀, t₁), δt) |> collect

B = zeros((nx, ny))
# Construct a grid that includes the initial Dome
η = 0.80
Δx = R₀ / nx / (η / 2)
Δy = R₀ / ny / (η / 2)
xs = [(i - nx / 2) * Δx for i in 1:nx]
ys = [(j - ny / 2) * Δy for j in 1:ny]

# Construct analytical ice thickness time series
Hs = [[halfar(x, y, t) for x in xs, y in ys] for t in tstops]
thicknessData = Sleipnir.ThicknessData(tstops, Hs)

# Construct analytical ice surface velocity time series
# For this, we use a shorter time threshold
tstops_vel = Huginn.define_callback_steps((t₀, t₁), 0.1) |> collect
Vs = [[halfar_velocity(x, y, t) for x in xs, y in ys] for t in tstops]
velocityData = Sleipnir.SurfaceVelocityData(
    x = xs, y = ys,
    vx = map(X -> getindex.(X, 1), Vs),
    vy = map(X -> getindex.(X, 1), Vs),
    vabs = map(X -> ODINN.norm.(X, 1), Vs),
    date = map(t -> Sleipnir.Dates.DateTime.(Sleipnir.partial_year(Sleipnir.Dates.Day, t)), tstops),
    isGridGlacierAligned = true
)

Hs_dome = map(x -> maximum(x), Hs)
# Reduction in ice thickness during simulation
frac_H = 100.0 * (Hs_dome[begin] - Hs_dome[end]) / Hs_dome[begin]
println("Maximum ice thickness has been reduce by $(frac_H) after $(Δt) years of forward simulation.%")
Rs_extent = map(x -> sum(x .> 0.0), Hs)
frac_R = 100.0 * (Rs_extent[end] - Rs_extent[begin]) / Rs_extent[begin]
println("Glacier extent has increased by $(frac_R)% after $(Δt) years of forward simulation.%")

params = Parameters(
    simulation = SimulationParameters(
        working_dir = working_dir,
        use_MB = use_MB,
        use_velocities = true,
        tspan = (t₀, t₁),
        step_MB = δt,
        multiprocessing = false,
        workers = 1,
        test_mode = false,
        rgi_paths = rgi_paths,
        gridScalingFactor = 1
    ),
    hyper = Hyperparameters(
        batch_size = 1,
        epochs = [20, 60],
        # epochs = 100,
        optimizer = [
            # ODINN.ADAM(0.001),
            ODINN.Optimisers.Adam(0.001, (0.0, 0.999)),
            # ODINN.GradientDescent(
            # linesearch = ODINN.LineSearches.BackTracking(iterations = 10)
            # ),
            # ODINN.LBFGS(
            #     linesearch = ODINN.LineSearches.BackTracking(iterations = 10),
            #     resetalpha = true
            #     ),
            ODINN.BFGS(
                linesearch = ODINN.LineSearches.BackTracking()
            )
        ]
    ),
    UDE = UDEparameters(
        sensealg = ZygoteAdjoint(),
        optim_autoAD = ODINN.NoAD(),
        grad = ContinuousAdjoint(
            abstol = 1e-6,
            reltol = 1e-6
        ),
        optimization_method = "AD+AD",
        target = :D        # empirical_loss_function = LossV(), # TODO
    ),
    solver = Huginn.SolverParameters(
        step = δt,
        progress = true
    )
)

# We are going to create a glacier using the Halfar solution
# TODO: Downscalling of glacier grid does not seem to be working
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
glaciers[1] = Glacier2D(
    glaciers[1],
    thicknessData = thicknessData    # velocityData = velocityData,
)

"""
We can define the architecture of the model directly, passing the prescale and postcale
directly to Lux using a WrappedFunction layer.
"""
# n_fourier_feautures = 10

function inv_normalize(v::Union{Vector, SubArray})
    @assert length(v) == 2
    return [
        ODINN.normalize(v[1]; lims = (0.0, H_max)), ODINN.normalize(v[2]; lims = (
            0.0, 0.6))]
end

# function fourier_feature(v::Union{Vector,SubArray})
#     return [fourier_feature(v[1], n_fourier_feautures); fourier_feature(v[2], n_fourier_feautures)]
# end

# Maximum value of U velocity for neural network
U₀ = 1e4
function post_scale(v::Union{Vector, SubArray})
    @assert length(v) == 1
    @assert 0.0 <= v[1] <= 1.0
    return [U₀ .* exp.((v[1] .- 1.0) ./ v[1])]
end

architecture = Lux.Chain(
    Lux.WrappedFunction(x -> LuxFunction(inv_normalize, x)),
    # WrappedFunction(x -> inv_fourier_feature(x)),
    # WrappedFunction(x -> [fourier_feature(x[1], n_fourier_feautures); fourier_feature(x[2], n_fourier_feautures)]),
    Lux.Dense(2, 5, x -> gelu.(x)),
    Lux.Dense(5, 8, x -> gelu.(x)),
    Lux.Dense(8, 20, x -> gelu.(x)),
    Lux.Dense(20, 30, x -> softplus.(x)),
    Lux.Dense(30, 10, x -> softplus.(x)),
    Lux.Dense(10, 1, sigmoid),
    # Lux.WrappedFunction(y -> U₀ .* exp.((y .- 1.0) ./ y))
    Lux.WrappedFunction(x -> LuxFunction(post_scale, x))    # WrappedFunction(y -> 10.0.^( 3.0 .* (y .- 1.0) .+ 1.0 .* (y .+ 1.0) ))
)

### Pretraining

pretrain = true
saved_nn_pretrain = true

if pretrain & !saved_nn_pretrain
    H_samples = 0.0:2.0:H_max
    ∇S_samples = 0.0:0.02:0.4
    all_combinations = vec(collect(Iterators.product(H_samples, ∇S_samples)))
    X_samples = hcat(first.(all_combinations), last.(all_combinations))
    function template_U(h, ∇s)
        (; ρ, g) = params.physical
        # Change parameters a little bit so we don't cheat that much
        n = 0.90 * n₀
        A = 1.6 * A₀
        # This one has one less H than the actual diffusivity
        return 2 * A * (ρ * g)^n * h^(n + 1) * ∇s^(n - 1) / (n + 2)
    end
    Y_samples = map(x -> template_U(x[1], x[2]), eachrow(X_samples))
    # Set matrices in right format
    X_samples = transpose(X_samples)

    # We will cap values that go above the maximum of the NN: 
    X_samples = X_samples[:, vec(Y_samples .< U₀)]
    Y_samples = Y_samples[Y_samples .< U₀]

    X_samples = Matrix(X_samples)
    Y_samples = reshape(Y_samples, (1, :))

    architecture, θ_pretrain,
    st_pretrain,
    losses = pretraining(
        architecture;
        X = X_samples, Y = Y_samples,
        nepochs = 5000, rng = rng
    )
    jldsave("./scripts/MWEs/inversion_diffusivity/data/pretrained.jld2"; θ = θ_pretrain)
elseif pretrain & saved_nn_pretrain
    # We read pretrained parameters of NN from memory
    θ_pretrain = load(
        joinpath(ODINN.root_dir, "scripts/MWEs/inversion_diffusivity/data", "pretrained.jld2"), "θ")
else
    θ_pretrain, st_pretrain = Lux.setup(rng, architecture)
end

# We define the prescale and postscale of quantities.
nn_model = NeuralNetwork(
    params;
    architecture = architecture,
    θ = ComponentVector(θ = θ_pretrain),
    seed = rng
)

# Define the law without pre and postscale.
# However, this means we need to specify the float type!
law = LawU(
    nn_model,
    params;
    prescale_bounds = nothing,
    max_NN = nothing,
    precompute_VJPs = true,
    precompute_interpolation = true
)

model = Model(
    iceflow = SIA2Dmodel(params; U = law),
    mass_balance = TImodel1(
        params; DDF = 6.0/1000.0,
        acc_factor = 1.2/1000.0
    ),
    regressors = (; U = nn_model),
    target = SIA2D_D_target(
        interpolation = :Linear,
        n_interp_half = 60
    )
)

# We create an ODINN prediction
functional_inversion = Inversion(model, glaciers, params)
