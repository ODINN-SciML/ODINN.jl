

function pde_solve_test()

    working_dir = joinpath(homedir(), "Python/ODINN_tests")
    oggm_config(working_dir)

    ## Retrieving gdirs and climate for the following glaciers  
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170",
    "RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570", "RGI60-04.07051",                	
    "RGI60-07.00274", "RGI60-07.01323", "RGI60-03.04207", "RGI60-03.03533", "RGI60-01.17316"]

    gdirs = init_gdirs(rgi_ids)
    tspan = (2010.0, 2015.0) # period in years for simulation
    gdirs_climate, gdirs_climate_batches = get_gdirs_with_climate(gdirs, tspan, overwrite=false, plot=false)

    # Load reference values for the simulation
    PDE_refs = load(joinpath(ODINN.root_dir, "data/PDE_refs.jld2"))["gdir_refs"]

    # Run the forward PDE simulation
    gdir_refs = @time generate_ref_dataset(gdirs_climate, tspan)

    let PDE_refs=PDE_refs
    for gdir_ref in gdir_refs
        let PDE_refs=PDE_refs, test_ref
        for PDE_ref in PDE_refs
            if gdir_ref["RGI_ID"] == PDE_ref["RGI_ID"]
                test_ref = PDE_ref
            end
        end

        # display(ODINN.Plots.heatmap(gdir_ref["H"] .- test_ref["H"], title="gdir_ref - test_ref"))

        @test gdir_ref["H"] ≈ test_ref["H"]
        @test gdir_ref["Vx"] ≈ test_ref["Vx"]
        @test gdir_ref["Vy"] ≈ test_ref["Vy"]
        end
    end
    end
end

pde_solve_test()
