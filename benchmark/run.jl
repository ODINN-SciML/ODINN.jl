import Pkg
Pkg.activate(dirname(Base.current_project()))

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
        velocities=true,
        tspan=tspan,
        step=δt,
        multiprocessing=false,
        workers=1,
        test_mode=true,
        rgi_paths=rgi_paths),
    UDE = UDEparameters(
        sensealg=SciMLSensitivity.ZygoteAdjoint(),
        optim_autoAD=ODINN.NoAD(),
        grad=DiscreteAdjoint(VJP_method=ODINN.EnzymeVJP()),
        optimization_method="AD+AD",
        target = "A"),
    solver = Huginn.SolverParameters(
        step=δt,
        save_everystep=true,
        progress=true)
)

model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = nothing,
    machine_learning = NeuralNetwork(params)
)

glaciers = initialize_glaciers(rgi_ids, params)

glacier_idx = 1
batch_idx = 1
H = glaciers[glacier_idx].H₀
simulation = FunctionalInversion(model, glaciers, params)
initialize_iceflow_model!(model.iceflow[batch_idx], glacier_idx, glaciers[glacier_idx], params)
t = tspan[1]
θ = simulation.model.machine_learning.θ
simulation.model.iceflow[batch_idx].glacier_idx = glacier_idx
λ = rand(size(H)...)

for VJPMode in (ODINN.EnzymeVJP(), ODINN.DiscreteVJP(), ODINN.ContinuousVJP())
    println("## Benchmark of $(VJPMode)")
    println("")
    println("<details>")
    println("")
    println("### VJP wrt H")
    trial = @benchmark ODINN.VJP_λ_∂SIA∂H($VJPMode, $λ, $H, $θ, $simulation, $t, $batch_idx)
    display(trial)
    println("")
    println("### VJP wrt θ")
    trial = @benchmark ODINN.VJP_λ_∂SIA∂θ($VJPMode, $λ, $H, $θ, $(nothing), $λ, $simulation, $t, $batch_idx)
    display(trial)
    println("")
    println("</details>")
    println("")
end
