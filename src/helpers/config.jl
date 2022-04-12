export initialize_ODINN

"""
    Initialize_ODINN(processes, python_path)

Initializes ODINN by configuring PyCall based on a given Python path. It also configures multiprocessing
for a given number of processes. 
"""
function initialize_ODINN(procs, python_path)
    
    ################################################
    ############  PYTHON ENVIRONMENT  ##############
    ################################################

    # Create structural folders if needed
    OGGM_path = joinpath(homedir(), "Python/OGGM_data")
    if !isdir(OGGM_path)
        mkpath(OGGM_path)
    end

    ## Set up Python environment
    global ENV["PYTHON"] = python_path 

    # @eval begin
    #     using PyCall
    #     Pkg.build("PyCall") 
    #     include(joinpath(ODINN.root_dir, "src/helpers/pycall.jl"))
    #     include(joinpath(ODINN.root_dir, "src/helpers/climate.jl"))
    #     include(joinpath(ODINN.root_dir, "src/helpers/oggm.jl"))
    # end # @eval

    if procs > 1
        if nprocs() < procs
            global processes = procs
            @eval begin
            addprocs(processes - nprocs(); exeflags="--project")
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            end # @eval
        end
         
        @eval begin
            @everywhere begin 
            using ODINN, PyCall, Infiltrator
            @eval ODINN begin
                ### PyCall configuration and Python libraries  ###
                include(joinpath(ODINN.root_dir, "src/helpers/pycall.jl"))
                ### Climate data processing  ###
                include(joinpath(ODINN.root_dir, "src/helpers/climate.jl"))
                ### OGGM configuration settings  ###
                include(joinpath(ODINN.root_dir, "src/helpers/oggm.jl"))
            end # @eval ODINN
            end # @everywhere
        end # @eval  
    end

end