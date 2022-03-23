export initialize_ODINN

"""
    Initialize_ODINN(processes, python_path)

Initializes ODINN by configuring PyCall based on a given Python path. It also configures multiprocessing
for a given number of processes. 
"""
function initialize_ODINN(processes, python_path)
    
    ################################################
    ############  PYTHON ENVIRONMENT  ##############
    ################################################

    ## Set up Python environment
    # Choose own Python environment with OGGM's installation
    # Use same path as "which python" in shell
    global ENV["PYTHON"] = python_path 

    @eval begin
    Pkg.build("PyCall") 
    include(joinpath(ODINN.root_dir, "src/helpers/pycall.jl"))
    include(joinpath(ODINN.root_dir, "src/helpers/climate.jl"))
    include(joinpath(ODINN.root_dir, "src/helpers/oggm.jl"))
    end # @eval

    if processes > 1
        if nprocs() < processes
            addprocs(processes - nprocs(); exeflags="--project")
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
        end
        
        @everywhere begin   
        @eval ODINN begin
        import Pkg
        using ODINN, Infiltrator
        ### PyCall configuration and Python libraries  ###
        include(joinpath(ODINN.root_dir, "src/helpers/pycall.jl"))
        ### Climate data processing  ###
        include(joinpath(ODINN.root_dir, "src/helpers/climate.jl"))
        ### OGGM configuration settings  ###
        include(joinpath(ODINN.root_dir, "src/helpers/oggm.jl"))
        end # @eval
        end # @everywhere
    end

end