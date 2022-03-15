export initialize

function initialize(processes, python_path)
    
    ################################################
    ############  PYTHON ENVIRONMENT  ##############
    ################################################

    ## Set up Python environment
    # Choose own Python environment with OGGM's installation
    # Use same path as "which python" in shell
    global ENV["PYTHON"] = python_path 

    @eval begin
    Pkg.build("PyCall") 
    ### PyCall configuration and Python libraries  ###
    include("helpers/pycall.jl")
    ### Climate data processing  ###
    include("helpers/climate.jl")
    ### OGGM configuration settings  ###
    include("helpers/oggm.jl")
    end # @eval

    if processes > 1
        if nprocs() < processes
            addprocs(processes - nprocs(); exeflags="--project")
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
        end

        @everywhere begin    
        @eval begin
        import Pkg
        Pkg.activate(dirname(Base.current_project()))
        using ODINN, Infiltrator
        ### PyCall configuration and Python libraries  ###
        include("helpers/pycall.jl")
        ### Climate data processing  ###
        include("helpers/climate.jl")
        ### OGGM configuration settings  ###
        include("helpers/oggm.jl")
        end # @eval
        end # @everywhere
    end

end