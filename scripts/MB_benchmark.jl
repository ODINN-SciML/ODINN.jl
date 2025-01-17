################################################
############  PYTHON ENVIRONMENT  ##############
################################################

import Pkg 
Pkg.activate(dirname(Base.current_project()))

using Distributed

@everywhere begin

using Revise
# using AbbreviatedStackTraces
using ODINN
using OrdinaryDiffEq, ODEInterfaceDiffEq, Sundials
using Optim, Optimization, OptimizationOptimJL, SciMLSensitivity
import OptimizationOptimisers.Adam
using JLD2
using BenchmarkTools, TimerOutputs, ProfileCanvas
using Infiltrator

using Dates

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

end # @everywhere

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

# Create a TimerOutput, this is the main type that keeps track of everything.
const to = TimerOutput()

# analysis = "types"
# analysis = "profile"
analysis = "timeit"

###############################################################
###########################  MAIN #############################
###############################################################


function run_benchmark(analysis)

    # We enable multiprocessing
    processes = 1
    ODINN.enable_multiprocessing(processes)
    
    # Flags
    @timeit to "setters" begin
    ODINN.set_use_MB(true) 
    # Spin up and reference simulations
    ODINN.set_run_spinup(false) # Run the spin-up simulation
    ODINN.set_use_spinup(false) # Use the updated spinup
    ODINN.set_create_ref_dataset(false) # Generate reference data for UDE training
    # UDE training
    ODINN.set_train(true)    # Train UDE 
    end

    tspan = (2010.0,2015.0) # period in years for simulation

    #########################################
    ###########  CLIMATE DATA  ##############
    #########################################

    if analysis == "timeit"

        # Defining glaciers to be modelled with RGI IDs
        # RGI60-11.03638 # Argentière glacier
        # RGI60-11.01450 # Aletsch glacier
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

        ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
        gdirs = @timeit to "init_gdirs" init_gdirs(rgi_ids)

        gdir_refs = @timeit to "PDE" @time generate_ref_dataset(gdirs, tspan; solver=RDPK3Sp35())
        println("Saving reference data")
        jldsave(joinpath(ODINN.root_dir, "data/gdir_refs.jld2"); gdir_refs)

        gdir_refs = load(joinpath(ODINN.root_dir, "data/gdir_refs.jld2"))["gdir_refs"]

        ODINN.reset_epochs()
        n_ADAM = 1
        batch_size = length(gdirs)
        UDE_settings = Dict("reltol"=>1e-7,
                            "solver"=>RDPK3Sp35(),
                            "sensealg"=>InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        ## First train with ADAM to move the parameters into a favourable space
        println("Training from scratch...")
        train_settings = (Adam(0.01), n_ADAM, batch_size) # optimizer, epochs
        # train_settings = (BFGS(initial_stepnorm=0.001), n_BFGS, batch_size) # optimizer, epochs
        iceflow_trained, UA_f, loss_history = @timeit to "UDE" train_iceflow_UDE(gdirs, gdir_refs,
                                                                        tspan, train_settings;
                                                                        UDE_settings=UDE_settings) 
        θ_trained = iceflow_trained.minimizer
        println("Saving NN weights...")
        jldsave(joinpath(ODINN.root_dir, "data/trained_weights.jld2"); θ_trained, ODINN.current_epoch)

        @show to
        @show ODINN.to

    elseif analysis == "profile"

        # Defining glaciers to be modelled with RGI IDs
        # RGI60-11.03638 # Argentière glacier
        # RGI60-11.01450 # Aletsch glacier
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

        ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
        gdirs = init_gdirs(rgi_ids)

        @profview_allocs generate_ref_dataset(gdirs, tspan; solver=RDPK3Sp35())

        # ODINN.reset_epochs()
        # n_ADAM = 1
        # batch_size = length(gdirs)
        # UDE_settings = Dict("reltol"=>1e-7,
        #                         "solver"=>RDPK3Sp35(),
        #                         "sensealg"=>InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), checkpointing=true))
        # ## First train with ADAM to move the parameters into a favourable space
        # println("Training from scratch...")
        # train_settings = (Adam(0.01), n_ADAM, batch_size) # optimizer, epochs
        # # train_settings = (BFGS(initial_stepnorm=0.001), n_BFGS, batch_size) # optimizer, epochs
        # iceflow_trained, UA_f, loss_history = @profview_allocs train_iceflow_UDE(gdirs, gdir_refs,
        #                                                                 tspan, train_settings;
        #                                                                 UDE_settings=UDE_settings)

    elseif analysis == "types"

        # Defining glaciers to be modelled with RGI IDs
        # RGI60-11.03638 # Argentière glacier
        # RGI60-11.01450 # Aletsch glacier
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

        ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
        gdirs = init_gdirs(rgi_ids)

        gdir_refs = generate_ref_dataset(gdirs, tspan; solver=RDPK3Sp35())

        ODINN.reset_epochs()
        n_ADAM = 1
        batch_size = length(gdirs)
        UDE_settings = Dict("reltol"=>1e-7,
                                "solver"=>RDPK3Sp35(),
                                "sensealg"=>InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), checkpointing=true))
        ## First train with ADAM to move the parameters into a favourable space
        println("Training from scratch...")
        train_settings = (Adam(0.01), n_ADAM, batch_size) # optimizer, epochs
        # train_settings = (BFGS(initial_stepnorm=0.001), n_BFGS, batch_size) # optimizer, epochs
        iceflow_trained, UA_f, loss_history = @code_warntype train_iceflow_UDE(gdirs, gdir_refs,
                                                                        tspan, train_settings;
                                                                        UDE_settings=UDE_settings)
    end

end

run_benchmark(analysis)



