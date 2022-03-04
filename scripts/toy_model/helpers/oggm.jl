###############################
####  OGGM configuration  #####
###############################

"""
    oggm_config()

Retrieves the initial glacier geometry (bedrock + ice thickness) for a glacier.
"""
function oggm_config(home_dir = cd(pwd, "../../../../.."), multiprocessing=true)
    cfg.initialize() # initialize OGGM configuration

    global PATHS = PyDict(cfg."PATHS")  # OGGM PATHS
    PATHS["working_dir"] = joinpath(home_dir, "Python/OGGM_data")  # Choose own custom path for the OGGM data
    global PARAMS = PyDict(cfg."PARAMS")
    PARAMS["hydro_month_nh"]=1

    # Multiprocessing 
    # PARAMS["prcp_scaling_factor"], PARAMS["ice_density"], PARAMS["continue_on_error"]
    global PARAMS["use_multiprocessing"] = multiprocessing # Let's use multiprocessing for OGGM
end

function init_gdirs(rgi_ids)
    # Retrieve glacier gdirs if they are available
    if length(readdir(joinpath(PATHS["working_dir"], "per_glacier"))) > 0
        gdirs = workflow.init_glacier_directories(rgi_ids)
    end
    # Generate all gdirs if needed
    if (!@isdefined gdirs) || ((@isdefined gdirs) && !isfile(joinpath(gdirs[1].dir, "gridded_data.nc")))  # check if data is already present
        gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=40, reset=false)
    
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
    end

    return gdirs
end


