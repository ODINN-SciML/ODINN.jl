###############################
####  OGGM configuration  #####
###############################

@everywhere using ParallelDataTransfer

export oggm_config, init_gdirs, PARAMS, PATHS

"""
    oggm_config()

Retrieves the initial glacier geometry (bedrock + ice thickness) for a glacier.
"""
function oggm_config()
    # sendto(workers(), multiprocessing=multiprocessing)
    @everywhere begin
    cfg.initialize() # initialize OGGM configuration
    
    global PATHS = PyDict(cfg."PATHS")  # OGGM PATHS
    PATHS["working_dir"] = joinpath(homedir(), "Python/OGGM_data")  # Choose own custom path for the OGGM data
    global PARAMS = PyDict(cfg."PARAMS")
    PARAMS["hydro_month_nh"]=1

    # Multiprocessing 
    # PARAMS["prcp_scaling_factor"], PARAMS["ice_density"], PARAMS["continue_on_error"]
    PARAMS["use_multiprocessing"] = true # Let's use multiprocessing for OGGM
    end # @everywhere

    # if multiprocessing
    #     @everywhere cfg.initialize()
    #     sendto(workers(), PATHS=PATHS, PARAMS=PARAMS)
    # end
end

"""
    init_gdirs(rgi_ids; force=false)

Initializes Glacier Directories using OGGM. Wrapper function calling `init_gdirs_scratch(rgi_ids)`.
"""
function init_gdirs(rgi_ids; force=false)
    # Try to retrieve glacier gdirs if they are available
    try
        if force
            gdirs = init_gdirs_scratch(rgi_ids)
        else
            gdirs = workflow.init_glacier_directories(rgi_ids)
        end
        return gdirs
    catch 
        @warn "Cannot retrieve gdirs from disk."
        println("Generating gdirs from scratch.")
        global create_ref_dataset = true # we force the creation of the reference dataset
        # Generate all gdirs if needed
        gdirs = init_gdirs_scratch(rgi_ids)
        return gdirs
    end
end

"""
    init_gdirs_scratch(rgi_ids)

Initializes Glacier Directories from scratch using OGGM.
"""
function init_gdirs_scratch(rgi_ids)
    # Check if some of the gdirs is missing files
    gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=40)
    list_talks = [
        # tasks.glacier_masks,
        # tasks.compute_centerlines,
        # tasks.initialize_flowlines,
        # tasks.compute_downstream_line,
        tasks.catchment_area,
        tasks.gridded_attributes,
        tasks.gridded_mb_attributes,
        # tasks.prepare_for_inversion,  # This is a preprocessing task
        # tasks.mass_conservation_inversion,  # This gdirsdoes the actual job
        # tasks.filter_inversion_output,  # This smoothes the thicknesses at the tongue a little
        # tasks.distribute_thickness_per_altitude,
        bedtopo.add_consensus_thickness   # Use consensus ice thicknesses from Farinotti et al. (2019)
    ]
    for task in list_talks
        # The order matters!
        workflow.execute_entity_task(task, gdirs)
    end
    return gdirs
end

