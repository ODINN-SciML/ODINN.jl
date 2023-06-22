
  struct OGGMparameters
    working_dir::String
    paths::PyDict
    params::PyDict
    multiprocessing::Bool
    workers::Int64
    ice_thickness_source::String
    base_url::String
end

"""
    OGGMparameters(;
        working_dir::String = joinpath(homedir(), "OGGM/OGGM_data"),
        paths::PyDict = nothing,
        params::PyDict = nothing,
        multiprocessing::Bool = false,
        workers::Int64 = 1,
        base_url::String = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands/"
        )
Initializes OGGM and it configures its parameters.
Keyword arguments
=================
    - `working_dir`: Working directory were all the files will be stored.
    - `paths`: Dictionary for OGGM-related paths.
    - `params`: Dictionary for OGGM-related parameters.
    - `multiprocessing`: Determines if multiprocessing is used for OGGM.
    - `workers`: How many workers are to be used for OGGM multiprocessing.
    - `ice_thickness_source`: Source for the ice thickness dataset. Either `Millan22` of `Farinotti19`.
    - `base_url`: Base URL to download all OGGM data.
"""
function OGGMparameters(;
            working_dir::String = joinpath(homedir(), "OGGM/OGGM_data"),
            paths::Union{PyDict, Nothing} = nothing,
            params::Union{PyDict, Nothing} = nothing,
            multiprocessing::Bool = false,
            workers::Int64 = 1,
            ice_thickness_source::String = "Millan22",
            base_url::String = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands/"
            )

    @assert ((ice_thickness_source == "Millan22") || (ice_thickness_source == "Farinotti19")) "Wrong ice thickness source! Should be either `Millan22` or `Farinotti19`."
    # initialize OGGM configuration
    cfg.initialize()
    
    if isnothing(paths)
        paths = PyDict(cfg."PATHS")  # OGGM PATHS
        paths["working_dir"] = working_dir # Choose own custom path for the OGGM data
    end
    if isnothing(params)
        params = PyDict(cfg."PARAMS")
        params["hydro_month_nh"]=1
        params["dl_verify"] = false
        params["continue_on_error"] = true # avoid stopping when a task fails for a glacier (e.g. lack of data)
        
        # Multiprocessing 
        params["use_multiprocessing"] = multiprocessing # Let's use multiprocessing for OGGM
        if multiprocessing
            params["mp_processes"] = workers
        end

    end

    # Build the OGGM parameters and configuration
    OGGM_parameters = OGGMparameters(working_dir, paths, params,
                                    multiprocessing, workers, 
                                    ice_thickness_source,
                                    base_url)

    return OGGM_parameters
end

"""
    oggm_config()

Configures the basic paths and parameters for OGGM.
"""
function oggm_config(working_dir=joinpath(homedir(), "OGGM/OGGM_data"); oggm_processes=1)
    @eval begin
    @everywhere begin
    @eval ODINN begin
    cfg.initialize() # initialize OGGM configuration
    
    PATHS = PyDict(cfg."PATHS")  # OGGM PATHS
    PATHS["working_dir"] = $working_dir # Choose own custom path for the OGGM data
    PARAMS = PyDict(cfg."PARAMS")
    PARAMS["hydro_month_nh"]=1
    PARAMS["dl_verify"] = false
    PARAMS["continue_on_error"] = true # avoid stopping when a task fails for a glacier (e.g. lack of data)

    # Multiprocessing 
    multiprocessing = $oggm_processes > 1 ? true : false
    PARAMS["use_multiprocessing"] = multiprocessing # Let's use multiprocessing for OGGM
    if multiprocessing
        PARAMS["mp_processes"] = $oggm_processes
    end



    end # @eval ODINN
    end # @everywhere
    end # @eval

    @eval ODINN begin

    end # @eval ODINN

end