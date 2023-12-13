
function __init__()
    if myid() == 1

        # Create structural folders if needed
        OGGM_path = joinpath(homedir(), "Python/OGGM_data")
        if !isdir(OGGM_path)
            mkpath(OGGM_path)
        end

    end

    try
        # Only load Python packages if not previously loaded by Sleipnir
        if cfg == PyNULL() && workflow == PyNULL() && utils == PyNULL() && MBsandbox == PyNULL() 
            println("Initializing Python libraries in ODINN...")
            # Load Python packages
            copy!(netCDF4, pyimport("netCDF4"))
            copy!(cfg, pyimport("oggm.cfg"))
            copy!(utils, pyimport("oggm.utils"))
            copy!(workflow, pyimport("oggm.workflow"))
            copy!(tasks, pyimport("oggm.tasks"))
            copy!(global_tasks, pyimport("oggm.global_tasks"))
            copy!(graphics, pyimport("oggm.graphics"))
            copy!(bedtopo, pyimport("oggm.shop.bedtopo"))
            copy!(millan22, pyimport("oggm.shop.millan22"))
            copy!(MBsandbox, pyimport("MBsandbox.mbmod_daily_oneflowline"))
            copy!(salem, pyimport("salem"))
            copy!(pd, pyimport("pandas"))
            copy!(np, pyimport("numpy"))
            copy!(xr, pyimport("xarray"))
            copy!(rioxarray, pyimport("rioxarray"))
        end
    catch e
        @warn "It looks like you have not installed and/or activated the virtual Python environment. \n 
        Please follow the guidelines in: https://github.com/ODINN-SciML/ODINN.jl#readme"
        @warn exception=(e, catch_backtrace())
    end
end


function clean()
    atexit() do
        run(`$(Base.julia_cmd())`)
    end
    exit()
 end

"""
    Initialize_ODINN(processes, python_path)

Initializes ODINN by configuring PyCall based on a given Python path. It also configures multiprocessing
for a given number of processes. 
"""
function enable_multiprocessing(params::Sleipnir.Parameters)
    procs::Int = params.simulation.workers
    if procs > 0 && params.simulation.multiprocessing
        if nprocs() < procs
            @eval begin
            addprocs($procs - nprocs(); exeflags="--project")
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            @everywhere using Reexport
            @everywhere @reexport using ODINN
            end # @eval
        elseif nprocs() != procs && procs == 1
            @eval begin
            rmprocs(workers(), waitfor=0)
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            end # @eval
        end
    end
    return nworkers()
end
