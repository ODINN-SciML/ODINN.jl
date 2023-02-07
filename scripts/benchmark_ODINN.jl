################################################
############  PYTHON ENVIRONMENT  ##############
################################################

import Pkg 
Pkg.activate(dirname(Base.current_project()))

using Distributed

@everywhere begin

using Revise
using AbbreviatedStackTraces
using ODINN
using OrdinaryDiffEq, ODEInterfaceDiffEq, Sundials
using Optim, Optimization, OptimizationOptimJL, SciMLSensitivity
import OptimizationOptimisers.Adam
using JLD2
using BenchmarkTools
using Infiltrator

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

end # @everywhere

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

# We enable multiprocessing
processes = 15
ODINN.enable_multiprocessing(processes)

###############################################################
###########################  MAIN #############################
###############################################################


function run_benchmark()

    function benchmark_setting(setting, UDE_settings, ude_benchmark,
        gdirs_climate, gdirs_climate_batches, gdir_refs,
        tspan, train_settings, θ_bm, random_MB; setting_name="solver")
    
        @assert (setting_name == "solver" || setting_name == "sensealg") "Wrong setting for the benchmark! Needs to be either `solver` or `sensealg`."
    
        ODINN.reset_epochs()
    
        if setting_name == "solver"
            UDE_settings["solver"] = setting
        elseif setting_name == "sensealg"
            UDE_settings["sensealg"] = setting
        end
    
        println("Benchmarking UDE settings: ", UDE_settings)
    
        # @benchmark train_iceflow_UDE($gdirs_climate, $tspan, $train_settings, $gdir_refs, $θ_bm, $UDE_settings)
    
        # @benchmark train_iceflow_UDE($gdirs_climate, $tspan, $train_settings, $gdir_refs, $θ_bm, $UDE_settings; random_MB=$random_MB)
    
        t_stats = @timed train_iceflow_UDE(gdirs_climate, gdirs_climate_batches, gdir_refs,
                                            tspan, train_settings, θ_bm;
                                            UDE_settings=UDE_settings,
                                            random_MB=random_MB) 

        # try
        #     t_stats = @timed train_iceflow_UDE(gdirs_climate, gdirs_climate_batches, gdir_refs,
        #                                                     tspan, train_settings, θ_bm;
        #                                                     UDE_settings=UDE_settings,
        #                                                     random_MB=random_MB) 
        #     # Save stats for each solver
        #     push!(ude_benchmark["time_stats"], t_stats)
        #     push!(ude_benchmark["ude_settings"], UDE_settings)
    
        # catch error
        #     println("ERROR: ", error)
        #     @warn "Solver not working. Skipping..."
        # end
    
        GC.gc()
        return ude_benchmark
    
    end

    # Flags
    ODINN.set_use_MB(true) 
    # Spin up and reference simulations
    ODINN.set_run_spinup(false) # Run the spin-up simulation
    ODINN.set_use_spinup(false) # Use the updated spinup
    ODINN.set_create_ref_dataset(false) # Generate reference data for UDE training
    # UDE training
    ODINN.set_train(true)    # Train UDE 

    tspan = (2010.0,2015.0) # period in years for simulation

    # Configure OGGM settings in all workers
    # Use a separate working dir to avoid conflicts with other simulations
    working_dir = joinpath(homedir(), "Python/OGGM_data_benchmark")
    oggm_config(working_dir)

    # Defining glaciers to be modelled with RGI IDs
    # RGI60-11.03638 # Argentière glacier
    # RGI60-11.01450 # Aletsch glacier
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

    ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
    gdirs = init_gdirs(rgi_ids)

    glacier_filter = 1
    gdir = [gdirs[glacier_filter]]

    #########################################
    ###########  CLIMATE DATA  ##############
    #########################################

    # Process climate data for glaciers
    gdirs_climate, gdirs_climate_batches = get_gdirs_with_climate(gdirs, tspan, overwrite=false, plot=false)

    # Generate random mass balance series for toy model
    if ODINN.use_MB[]
        random_MB = generate_random_MB(gdirs_climate, tspan; plot=false)
    else
        random_MB = nothing
    end
    # Run forward model for selected glaciers
    if ODINN.create_ref_dataset
        println("Generating reference dataset for training...")
        solver = RDPK3Sp35()
        # solver = Ralston()
        # Compute reference dataset in parallel
        gdir_refs = @time generate_ref_dataset(gdirs_climate, tspan; solver=solver, random_MB=random_MB)

        println("Saving reference benchmark data")
        jldsave(joinpath(ODINN.root_dir, "data/PDE_refs_benchmark.jld2"); gdir_refs)
    end

    # Load stored PDE reference datasets
    gdir_refs = load(joinpath(ODINN.root_dir, "data/PDE_refs_benchmark.jld2"))["gdir_refs"]

    #######################################################################################################
    #############################             Train UDE            ########################################
    #######################################################################################################

    θ_bm = load(joinpath(ODINN.root_dir, "data/benchmark_weights.jld"))["θ_benchmark"]
    
    # Solvers 
    # ude_solvers = [Ralston(), CKLLSRK54_3C(), RDPK3Sp35(), RDPK3SpFSAL35(), ROCK4()]
    solver = RDPK3Sp35()
    ude_solvers = [OwrenZen3(), VCABM(), Vern6(), AN5(), AB3(), KenCarp3(), TRBDF2(), ROCK4(), RDPK3Sp35(), CKLLSRK54_3C(), 
                    QNDF(autodiff=false), FBDF(autodiff=false)]
    # ude_solvers = [OwrenZen3(), VCABM(), Vern6(), AN5(), AB3(), KenCarp3(), TRBDF2(), ROCK4(), RDPK3Sp35(), CKLLSRK54_3C(), 
    #                 radau(), CVODE_BDF(), QNDF(autodiff=false), FBDF(autodiff=false)]
    ude_benchmark = Dict("ude_settings"=>[], "time_stats"=>[])

    # sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), checkpointing=true) # MB only compatible with ReverseDiffVJP() for now
    # sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP())

    # sensealgs = [InterpolatingAdjoint(autojacvec=ReverseDiffVJP()), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), checkpointing=true),
    #                 InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true), InterpolatingAdjoint(autojacvec=EnzymeVJP(), checkpointing=true),
    #                 QuadratureAdjoint(autojacvec=ReverseDiffVJP()), QuadratureAdjoint(autojacvec=ZygoteVJP()), QuadratureAdjoint(autojacvec=EnzymeVJP()), 
    #                 BacksolveAdjoint(autojacvec=ReverseDiffVJP(), checkpointing=true), BacksolveAdjoint(autojacvec=ZygoteVJP(), checkpointing=true)]

    sensealgs = [InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), checkpointing=true)]


    reltol = 1e-7
    opt = Adam(0.005)
    batch_size = length(gdir_refs)
    # opt = BFGS(initial_stepnorm=0.02f0)
    train_settings = (opt, 1, batch_size) # optimizer, epochs

    testing = "sensealg"
    # testing = "solver"

    if testing == "sensealg"
        UDE_settings = Dict("reltol"=>reltol,
                            "solver"=>solver,
                            "sensealg"=>[])

        # Benchmark every solver in parallel
        ude_benchmarks = map(sensealg -> benchmark_setting(sensealg, UDE_settings, ude_benchmark,
                                            gdirs_climate, gdirs_climate_batches, gdir_refs,
                                            tspan, train_settings, θ_bm, random_MB; setting_name="sensealg"), sensealgs)     
                                            
    elseif testing == "solver"
        UDE_settings = Dict("reltol"=>reltol,
        "solver"=>[],
        "sensealg"=>sensealg)

        # Benchmark every solver in parallel
        ude_benchmarks = pmap(ude_solver -> benchmark_setting(ude_solver, UDE_settings, ude_benchmark,
                                                gdirs_climate, gdirs_climate_batches, gdir_refs,
                                                tspan, train_settings, θ_bm, random_MB; setting_name="solver"), ude_solvers) 
                                            
    end
    # Save benchmark results
    println("Saving benchmark...")
    ODINN.jldsave(joinpath(ODINN.root_dir, "data/time_stats_benchmark_$testing.jld2"); ude_benchmarks) 

    # @show ude_benchmarks

    for ude_benchmark in ude_benchmarks
        if length(ude_benchmark["ude_settings"]) > 0
            if testing == "sensealg"
                println(ude_benchmark["ude_settings"]["sensealg"], " - ", ude_benchmark["time_stats"].time)
            elseif testing == "solver"
                println(ude_benchmark["ude_settings"]["solver"], " - ", ude_benchmark["time_stats"].time)
            end
        end
    end
end

run_benchmark()

ude_benchmarks = load("data/time_stats_benchmark_sensealg.jld2")["ude_benchmarks"]

for ude_benchmark in ude_benchmarks
    if length(ude_benchmark["ude_settings"]) > 0
        if !isnothing(ude_benchmark["ude_settings"][1]["sensealg"])
            println(ude_benchmark["ude_settings"][1]["sensealg"], " - ", ude_benchmark["time_stats"][1].time)
        end
    end
end


