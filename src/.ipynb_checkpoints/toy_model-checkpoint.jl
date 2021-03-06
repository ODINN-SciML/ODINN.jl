import ODINN

processes = 16
python_path = "/home/jovyan/.conda/envs/oggm_env/bin/python3.9"
ODINN.initialize(processes, python_path)

function run()
    # Configure OGGM settings in all workers
    @everywhere oggm_config()

    ###############################################################
    ###########################  MAIN #############################
    ###############################################################

    # Defining glaciers to be modelled with RGI IDs
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170",
                "RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570", "RGI60-04.07051",                	
                "RGI60-07.00274", "RGI60-07.01323", "RGI60-03.04207", "RGI60-03.03533", "RGI60-01.17316"]

    ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
    gdirs = init_gdirs(rgi_ids)

    #########################################
    ###########  CLIMATE DATA  ##############
    #########################################

    # Process climate data for glaciers
    gdirs_climate = get_gdirs_with_climate(gdirs, true)

    # Run forward model for selected glaciers
    if create_ref_dataset 
        println("Generating reference dataset for training...")
    
        # Compute reference dataset in parallel
        @everywhere solver = Ralston()
        H_refs, V̄x_refs, V̄y_refs = @time generate_ref_dataset(gdirs_climate)
            
        println("Saving reference data")
        jldsave(joinpath(root_dir, "data/PDE_refs.jld2"); H_refs, V̄x_refs, V̄y_refs)
    end

    # Load stored PDE reference datasets
    PDE_refs = load(joinpath(root_dir, "data/PDE_refs.jld2"))
    # Plot training dataset of glaciers
    plot_glacier_dataset(gdirs_climate, PDE_refs)

    #######################################################################################################
    #############################             Train UDE            ########################################
    #######################################################################################################

    # Training setup
    global current_epoch = 1

    # Train iceflow UDE in parallel
    # First train with ADAM to move the parameters into a favourable space
    @everywhere solver = ROCK4()

    if retrain
        println("Retraining from previous NN weights...")
        trained_weights = load(joinpath(root_dir, "data/trained_weights.jld2"))
        current_epoch = trained_weights["current_epoch"]
        θ_trained = trained_weights["θ_trained"]
        # train_settings = (BFGS(initial_stepnorm=0.05), 20) # optimizer, epochs
        train_settings = (ADAM(0.01), 20) # optimizer, epochs
        iceflow_trained, UA = @time train_iceflow_UDE(gdirs_climate, train_settings, PDE_refs, θ_trained) # retrain
        θ_trained = iceflow_trained.minimizer

        # Save trained NN weights
        println("Saving NN weights...")
        jldsave(joinpath(root_dir, "data/trained_weights.jld2"); θ_trained, current_epoch)
    else
        println("Training from scratch...")
        train_settings = (ADAM(0.05), 10) # optimizer, epochs
        iceflow_trained, UA = @time train_iceflow_UDE(gdirs_climate, train_settings, PDE_refs)
        θ_trained = iceflow_trained.minimizer
        println("Saving NN weights...")
        jldsave(joinpath(root_dir, "data/trained_weights.jld2"); θ_trained, current_epoch)

        # Continue training with BFGS
        # train_settings = (BFGS(initial_stepnorm=0.02f0), 20) # optimizer, epochs
        train_settings = (ADAM(0.02), 20) # optimizer, epochs
        iceflow_trained, UA = @time train_iceflow_UDE(gdirs_climate, train_settings, PDE_refs, θ_trained) # retrain
        θ_trained = iceflow_trained.minimizer
        # Save trained NN weights
        println("Saving NN weights...")
        jldsave(joinpath(root_dir, "data/trained_weights.jld2"); θ_trained, current_epoch)
    end

    ##########################################
    ####  Plot the final trained model  ######
    ##########################################
    data_range = -20.0:0.0
    pred_A = predict_A̅(UA, θ_trained, collect(data_range)')
    pred_A = [pred_A...] # flatten
    true_A = A_fake(data_range) 

    Plots.scatter(true_A, label="True A")
    train_final = Plots.plot!(pred_A, label="Predicted A")
    Plots.savefig(train_final,joinpath(root_plots,"training","final_model.png"))
end

# Run main
run()
