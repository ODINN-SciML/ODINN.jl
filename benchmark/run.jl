import Pkg
Pkg.activate(dirname(Base.current_project()))

# Use a fork of SciMLSensitivity until https://github.com/SciML/SciMLSensitivity.jl/issues/1238 is fixed
Pkg.develop(url="https://github.com/albangossard/SciMLSensitivity.jl/")

using ODINN
using BenchmarkTools
using Logging
Logging.disable_logging(Logging.Info)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
using Random
using SciMLSensitivity

println("# Performance benchmark")

Random.seed!(1234)

rgi_ids = ["RGI60-11.03638"]
rgi_paths = get_rgi_paths()
tspan = (2010.0, 2015.0)
δt = 1/12
params = Parameters(
    simulation = SimulationParameters(
        use_MB=false,
        use_velocities=true,
        tspan=tspan,
        step=δt,
        multiprocessing=false,
        test_mode=true,
        rgi_paths=rgi_paths),
    UDE = UDEparameters(
        sensealg=SciMLSensitivity.ZygoteAdjoint(),
        optim_autoAD=ODINN.NoAD(),
        grad=DiscreteAdjoint(VJP_method=ODINN.EnzymeVJP()),
        optimization_method="AD+AD",
        target = :A),
    solver = Huginn.SolverParameters(
        step=δt,
        save_everystep=true,
        progress=true)
)

nn_model = NeuralNetwork(params)
model = Model(
    iceflow = SIA2Dmodel(params; A=LawA(nn_model, params)),
    mass_balance = nothing,
    regressors = (; A=nn_model)
)

glaciers = initialize_glaciers(rgi_ids, params)

glacier_idx = 1
batch_idx = 1
H = glaciers[glacier_idx].H₀
simulation = FunctionalInversion(model, glaciers, params)
simulation.cache = init_cache(model, simulation, glacier_idx, params)
t = tspan[1]
θ = simulation.model.machine_learning.θ
λ = rand(size(H)...)

for VJPMode in (ODINN.EnzymeVJP(), ODINN.DiscreteVJP(), ODINN.ContinuousVJP())
    println("## Benchmark of $(VJPMode)")
    println("")
    println("<details>")
    println("")
    println("### VJP wrt H")
    trial = @benchmark ODINN.VJP_λ_∂SIA∂H($VJPMode, $λ, $H, $θ, $simulation, $t)
    display(trial)
    println("")
    println("### VJP wrt θ")
    trial = @benchmark ODINN.VJP_λ_∂SIA∂θ($VJPMode, $λ, $H, $θ, $(nothing), $simulation, $t)
    display(trial)
    println("")
    println("</details>")
    println("")
end
